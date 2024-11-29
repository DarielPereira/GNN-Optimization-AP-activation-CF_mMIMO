import numpy as np
import itertools
import numpy.linalg as linalg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, localScatteringR, correlationNormalized_grid
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink


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


def AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, Hhat, H, B, C, L, K, N, Q, M,
                   comb_mode, heuristic_mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    match heuristic_mode:
        case 'exhaustive_search':

            # compute all the feasible AP state vectors
            feasible_APstates = np.array(list(itertools.product([0, 1], repeat=L)))

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
                        print('ERROR: Combining mismatching')
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

                best_sum_SE = 0

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
                                print('ERROR: Combining mismatching')
                                SE = 0

                        sum_SE = np.sum(SE)

                        if sum_SE > best_sum_SE:
                            best_sum_SE = sum_SE
                            best_APstate_toAdd = APstate_toAdd

                # Update the best AP state
                best_APstate = best_APstate + best_APstate_toAdd

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
                    print('ERROR: Combining mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()

    return best_APstate, best_sum_SE, best_SEs











