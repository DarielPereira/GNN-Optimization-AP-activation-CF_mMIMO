"""
This script contains functions for pilot assignment and AP on/off strategies
"""


import numpy as np
import itertools
import torch as th

from functionsUtils import db2pow, binary_combinations
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsSetup import get_F_G_matrices
from functionsGraphHandling import bipartitegraph_generation
from functionsGraphHandling import GNN_CorrMat, GNN_Gains


def PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode):
    """Compute the pilot assignment for a set of UEs
    INPUT>
    :param R: matrix with dimensions (N, N, L, K) containing the channel correlation matrices
    :param gainOverNoisedB: matrix with dimensions (L, K) containing the channel gains
    :param tau_p: number of pilots
    :param L: number of APs
    :param K: number of UEs
    :param N: number of antennas at the APs
    :param mode: pilot assignment mode
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # to store pilot assignment
    pilotIndex = -1 * np.ones((K), int)

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'DCC':

            # Determine the pilot assignment
            for k in range(0, K):

                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                if k <= tau_p - 1:  # Assign orthogonal pilots to the first tau_p UEs
                    pilotIndex[k] = k

                else:  # Assign pilot for remaining users

                    # Compute received power to the master AP from each pilot
                    pilotInterference = np.zeros(tau_p)

                    for t in range(tau_p):
                        pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master, :k][pilotIndex[:k] == t]))

                    # Find the pilot with least received power
                    bestPilot = np.argmin(pilotInterference)
                    pilotIndex[k] = bestPilot

    return pilotIndex


def AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, Hhat, H, B, C, L, K, N, Q, M, f,
                   comb_mode, heuristic_mode, GNN_mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    match heuristic_mode:
        case 'bestgains_individualAPs':
            ave_gainOverNoisedB = np.mean(gainOverNoisedB, axis=1)

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Find the best AP state
            for c in range(M.shape[0]):
                filtered_ave_gainOverNoisedB = np.array(ave_gainOverNoisedB)
                filtered_ave_gainOverNoisedB[np.where(M[c, :] != 1)[0]] = float('-inf')
                indices = np.argsort(filtered_ave_gainOverNoisedB)[-Q:]
                best_APstate[indices] = 1

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()


        case 'exhaustive_search':

            # compute all the feasible AP state vectors
            # feasible_APstates = np.array(list(itertools.product([0, 1], repeat=L)))

            # Compute the feasible AP state vectors that consider the maximum number of active APs
            feasible_APstates = binary_combinations(L, Q*M.shape[0])

            # Compute the valid AP state vectors
            valid_APstates = []
            for feasible_APstate in feasible_APstates:
                valid = True
                for idx in range(M.shape[0]):
                    if M[idx, :] @ feasible_APstate > Q:
                        valid = False
                if valid:
                    valid_APstates.append(feasible_APstate)

            # To store the sum-SE values
            sum_SEs = np.zeros((len(valid_APstates)))

            # To store the individual SE values
            SEs = np.zeros((len(valid_APstates), K))

            # Try each valid AP state:
            for idx, APstate in enumerate(valid_APstates):
                # D vector common to all the UEs
                D = np.ones((L, K))

                # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                               nbrOfRealizations, N, K, L, p)

                match comb_mode:
                    case 'MMSE':
                        SE = SE_MMSE
                    case 'P_RZF':
                        SE = SE_P_RZF
                    case 'MR':
                        SE = SE_MR
                    case 'P_MMSE':
                        SE = SE_P_MMSE
                    case _:
                        print('ERROR: Combining mode mismatching')
                        SE = 0

                sum_SE = np.sum(SE)

                sum_SEs[idx] = sum_SE
                SEs[idx, :] = SE.flatten()

            # Find the best AP state
            best_APstate = valid_APstates[np.argmax(sum_SEs)]

            # Get the best sum-SE
            best_sum_SE = np.max(sum_SEs)

            # Get the best individual SEs
            best_SEs = SEs[np.argmax(sum_SEs), :]

        case 'sequential_greedy':

            # Compute all the feasible 1-AP state vectors
            unused_feasible_1AP_states = list(np.eye(L, dtype=int))

            # To store best AP state
            best_APstate = np.zeros((L), int)

            # Stop when no other AP can be added
            while sum(best_APstate) < M.shape[0]*Q:

                sum_SEs = []

                # Try to add an AP from those that have not been used
                for APstate_toAdd in unused_feasible_1AP_states:

                    # D vector common to all the UEs
                    D = np.ones((L, K))

                    APstate = best_APstate + APstate_toAdd

                    # Check for validity of the AP state
                    valid = True
                    for idx in range(M.shape[0]):
                        if M[idx, :] @ APstate > Q:
                            valid = False

                    if valid:

                        # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                        SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                                       nbrOfRealizations, N, K, L, p)

                        match comb_mode:
                            case 'MMSE':
                                SE = SE_MMSE
                            case 'P_RZF':
                                SE = SE_P_RZF
                            case 'MR':
                                SE = SE_MR
                            case 'P_MMSE':
                                SE = SE_P_MMSE
                            case _:
                                print('ERROR: Combining mode mismatching')
                                SE = 0

                        sum_SE = np.sum(SE)

                        sum_SEs.append((sum_SE, APstate_toAdd))

                if len(sum_SEs) == 0:
                    break
                # Update the best AP state
                best_APstate_toAdd = max(sum_SEs, key= lambda x: x[0])[1]
                best_APstate += best_APstate_toAdd

                # Remove the best AP state from the unused feasible 1-AP states
                # Find the index of the array to remove
                index_to_remove = next(i for i, state in enumerate(unused_feasible_1AP_states) if
                                       np.array_equal(state, best_APstate_toAdd))

                # Remove the array using the index
                unused_feasible_1AP_states.pop(index_to_remove)

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c, tau_p,
                                                                               nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'best_individualAPs':

            # Compute all the feasible 1-AP state vectors
            feasible_1AP_states = list(np.eye(L, dtype=int))

            # To store sum-SE values and the corresponding AP states
            sum_SEs = np.zeros((L))

            # Run over the AP states
            for idx, APstate in enumerate(feasible_1AP_states):

                # D vector common to all the UEs
                D = np.ones((L, K))

                # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                               nbrOfRealizations, N, K, L, p)

                match comb_mode:
                    case 'MMSE':
                        SE = SE_MMSE
                    case 'P_RZF':
                        SE = SE_P_RZF
                    case 'MR':
                        SE = SE_MR
                    case 'P_MMSE':
                        SE = SE_P_MMSE
                    case _:
                        print('ERROR: Combining mode mismatching')
                        SE = 0

                sum_SE = np.sum(SE)

                sum_SEs[idx] = sum_SE

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Find the best AP state
            for c in range(M.shape[0]):
                filtered_sum_SEs = np.array(sum_SEs)
                filtered_sum_SEs[np.where(M[c, :]!=1)[0]] = 0
                indices = np.argsort(filtered_sum_SEs)[-Q:]
                best_APstate[indices] = 1

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'local_ES':

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):
                # D vector common to all the UEs
                D = np.ones((L, K))

                # number of APs connected to CPU c
                T_c = sum(M[c, :])

                # Compute all the feasible combinations with exactly Q APs
                feasible_reduced_APstates = binary_combinations(T_c, Q)

                if len(feasible_reduced_APstates) == 0:
                    continue

                # To store the sum-SE values
                sum_SEs = np.zeros((len(feasible_reduced_APstates)))

                # To store the individual SE values
                SEs = np.zeros((len(feasible_reduced_APstates), K))

                # Try each valid AP state:
                for idx, reduced_APstate in enumerate(feasible_reduced_APstates):

                    APstate = np.zeros((L))
                    APstate[M[c, :]==1] = reduced_APstate

                    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                    SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                                   nbrOfRealizations, N, K, L, p)

                    match comb_mode:
                        case 'MMSE':
                            SE = SE_MMSE
                        case 'P_RZF':
                            SE = SE_P_RZF
                        case 'MR':
                            SE = SE_MR
                        case 'P_MMSE':
                            SE = SE_P_MMSE
                        case _:
                            print('ERROR: Combining mode mismatching')
                            SE = 0

                    sum_SE = np.sum(SE)

                    sum_SEs[idx] = sum_SE
                    SEs[idx, :] = SE.flatten()

                # Find the best AP state
                best_APstate[M[c, :]==1] = feasible_reduced_APstates[np.argmax(sum_SEs)]

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'local_SG':
            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):
                # D vector common to all the UEs
                D = np.ones((L, K))

                # number of APs connected to CPU c
                T_c = sum(M[c, :])

                # Compute all the feasible 1-AP state vectors
                unused_feasible_1AP_states = list(np.eye(T_c, dtype=int))

                # To store best AP state
                local_best_APstate = np.zeros((T_c), int)

                # Stop when no other AP can be added
                while sum(local_best_APstate) < Q:

                    sum_SEs = []

                    # Try to add an AP from those that have not been used
                    for APstate_toAdd in unused_feasible_1AP_states:

                        APstate = np.zeros((L))
                        APstate[M[c, :]==1] = local_best_APstate + APstate_toAdd

                        # D vector common to all the UEs
                        D = np.ones((L, K))

                        # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                        SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                                       nbrOfRealizations, N, K, L, p)

                        match comb_mode:
                            case 'MMSE':
                                SE = SE_MMSE
                            case 'P_RZF':
                                SE = SE_P_RZF
                            case 'MR':
                                SE = SE_MR
                            case 'P_MMSE':
                                SE = SE_P_MMSE
                            case _:
                                print('ERROR: Combining mode mismatching')
                                SE = 0

                        sum_SE = np.sum(SE)

                        sum_SEs.append((sum_SE, APstate_toAdd))

                    if len(sum_SEs) == 0:
                        break

                    # Update the best AP state
                    best_APstate_toAdd = max(sum_SEs, key= lambda x: x[0])[1]
                    local_best_APstate += best_APstate_toAdd

                    # Remove the best AP state from the unused feasible 1-AP states
                    # Find the index of the array to remove
                    index_to_remove = next(i for i, state in enumerate(unused_feasible_1AP_states) if
                                           np.array_equal(state, best_APstate_toAdd))

                    # Remove the array using the index
                    unused_feasible_1AP_states.pop(index_to_remove)

                # update the best local AP state in the global AP state
                best_APstate[M[c, :]==1] = local_best_APstate

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'Q_random':

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):

                # Get the APs connected to the CPU c
                connected_APs = np.where(M[c, :]==1)[0]

                # Randomly select Q APs
                selected_APs = np.random.choice(connected_APs, min(Q,len(connected_APs)) , replace=False)

                # Update the best AP state
                best_APstate[selected_APs] = 1

            # Check if the best AP state is valid
            valid = True
            for idx in range(M.shape[0]):
                if M[idx, :] @ best_APstate > Q:
                    valid = False

            if not valid:
                print('ERROR: Invalid AP state')

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'fixed':

            np.random.seed(1)

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):
                # Get the APs connected to the CPU c
                connected_APs = np.where(M[c, :] == 1)[0]

                # Randomly select Q APs
                selected_APs = np.random.choice(connected_APs, min(Q, len(connected_APs)), replace=False)

                # Update the best AP state
                best_APstate[selected_APs] = 1

            # Check if the best AP state is valid
            valid = True
            for idx in range(M.shape[0]):
                if M[idx, :] @ best_APstate > Q:
                    valid = False

            if not valid:
                print('ERROR: Invalid AP state')

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()


        case 'successive_local_SG':

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):
                # D vector common to all the UEs
                D = np.ones((L, K))

                # number of APs connected to CPU c
                T_c = sum(M[c, :])

                # Compute all the feasible 1-AP state vectors
                unused_feasible_1AP_states = list(np.eye(T_c, dtype=int))

                # To store best AP state
                local_best_APstate = np.zeros((T_c), int)

                # Stop when no other AP can be added
                while sum(local_best_APstate) < Q:

                    sum_SEs = []

                    # Try to add an AP from those that have not been used
                    for APstate_toAdd in unused_feasible_1AP_states:

                        APstate = best_APstate

                        APstate[M[c, :] == 1] = local_best_APstate + APstate_toAdd

                        # D vector common to all the UEs
                        D = np.ones((L, K))

                        # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                        SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c,
                                                                                       tau_p,
                                                                                       nbrOfRealizations, N, K, L, p)

                        match comb_mode:
                            case 'MMSE':
                                SE = SE_MMSE
                            case 'P_RZF':
                                SE = SE_P_RZF
                            case 'MR':
                                SE = SE_MR
                            case 'P_MMSE':
                                SE = SE_P_MMSE
                            case _:
                                print('ERROR: Combining mode mismatching')
                                SE = 0

                        sum_SE = np.sum(SE)

                        sum_SEs.append((sum_SE, APstate_toAdd))

                    if len(sum_SEs) == 0:
                        break

                    # Update the best AP state
                    best_APstate_toAdd = max(sum_SEs, key=lambda x: x[0])[1]
                    local_best_APstate += best_APstate_toAdd

                    # Remove the best AP state from the unused feasible 1-AP states
                    # Find the index of the array to remove
                    index_to_remove = next(i for i, state in enumerate(unused_feasible_1AP_states) if
                                           np.array_equal(state, best_APstate_toAdd))

                    # Remove the array using the index
                    unused_feasible_1AP_states.pop(index_to_remove)

                # update the best local AP state in the global AP state
                best_APstate[M[c, :] == 1] = local_best_APstate

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'successive_local_ES':

            # To store the best AP state
            best_APstate = np.zeros((L))

            # Run over the CPUs
            for c in range(M.shape[0]):
                # D vector common to all the UEs
                D = np.ones((L, K))

                # number of APs connected to CPU c
                T_c = sum(M[c, :])

                # Compute all the feasible combinations with exactly Q APs
                feasible_reduced_APstates = binary_combinations(T_c, Q)

                if len(feasible_reduced_APstates) == 0:
                    continue

                # To store the sum-SE values
                sum_SEs = np.zeros((len(feasible_reduced_APstates)))

                # To store the individual SE values
                SEs = np.zeros((len(feasible_reduced_APstates), K))

                # Try each valid AP state:
                for idx, reduced_APstate in enumerate(feasible_reduced_APstates):

                    APstate = best_APstate
                    APstate[M[c, :] == 1] = reduced_APstate

                    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                    SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, APstate, C, tau_c, tau_p,
                                                                                   nbrOfRealizations, N, K, L, p)

                    match comb_mode:
                        case 'MMSE':
                            SE = SE_MMSE
                        case 'P_RZF':
                            SE = SE_P_RZF
                        case 'MR':
                            SE = SE_MR
                        case 'P_MMSE':
                            SE = SE_P_MMSE
                        case _:
                            print('ERROR: Combining mode mismatching')
                            SE = 0

                    sum_SE = np.sum(SE)

                    sum_SEs[idx] = sum_SE
                    SEs[idx, :] = SE.flatten()

                # Find the best AP state
                best_APstate[M[c, :] == 1] = feasible_reduced_APstates[np.argmax(sum_SEs)]

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case 'GNN':

            # Get the F matrix with preferred APs for each UE
            F, G = get_F_G_matrices(gainOverNoisedB, L, K, f)

            # Store the graph information
            # Generate the list of edges in the graphs
            G_sameCPU = np.zeros((L, L), dtype=int)
            G_sameCPU_full = np.zeros((L, L), dtype=int)
            for c in range(M.shape[0]):
                G_sameCPU[np.where(M[c, :] == 1)[0], :] = G[np.where(M[c, :] == 1)[0], :] * M[c, :]
                G_sameCPU_full[np.where(M[c, :] == 1)[0], :] = M[c, :]

            G_sameCPU_full = G_sameCPU_full - np.identity(L)
            G_diffCPU = G - G_sameCPU

            G_sameCPU_graph = th.tensor(np.transpose(np.nonzero(G_sameCPU))).T
            G_sameCPU_fullgraph = th.tensor(np.transpose(np.nonzero(G_sameCPU_full))).T
            G_diffCPU_graph = th.tensor(np.transpose(np.nonzero(G_diffCPU))).T

            F_graph, UE_features = bipartitegraph_generation(F, R, gainOverNoisedB, GNN_mode)

            # Create the GNN
            match GNN_mode:
                case 'CorrMat':
                    GNN = GNN_CorrMat(UE_features.shape[1])
                case 'Gains':
                    GNN = GNN_Gains(UE_features.shape[1])
                case _:
                    raise ValueError('ERROR: GNN mode mismatching')

            GNN.load_model(f'./AP_TrainingData/' + GNN_mode +
                           '/Model_L_12_N_4_Q_2_T_4_f_5_taup_100_NbrSamp_20000_Epochs_7_SAGEConv_sum.pt')

            # Compute the prediction
            GNN_output = GNN(G_sameCPU_fullgraph, G_diffCPU_graph,
                                           UE_features, F_graph, L)

            APs_probabilities = th.sigmoid(GNN_output).detach().numpy().flatten()

            # To store the best AP state
            best_APstate = np.zeros((L))

            for c in range(M.shape[0]):
                best_APstate[np.argsort((APs_probabilities*M[c,:]).flatten())[-Q:]] = 1

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

        case _:
            print('ERROR: Heuristic mode mismatching')



    return best_APstate, best_sum_SE, best_SEs











