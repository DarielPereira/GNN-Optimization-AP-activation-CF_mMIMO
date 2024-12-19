# AP on/off sum-rate optimization for cell-free massive MIMO communications with limited fronthaul capacity

## Versions:
    # 20241129: 
        # Added:
            # Include the script AP_OnOff_Heuristics.py that calls the heuristic methods.
            # Update the functionsSetup.py to generate and return the M matrix that associates APs and CPUs.
            # Include the function AP_OnOff_GlobalHeuristics() in functionsAPAllocation.py to implement the global 
            heuristic methods:
                - 'exhaustive_search': the exhaustive search method (only feasible in setups with reduced number of APs).
                - 'sequential_greedy': incude an AP greedily in each step.

    # 20241212: 
        # Added:
            # Include the function AP_OnOff_GlobalHeuristics() in functionsAPAllocation.py to implement the global 
            heuristic methods:
                - 'best_individualAPs': select the best AP according the their individual performance.
                - 'local_ES': perform the exhaustive search locally for the APs in each CPU.
                - 'local_SG': perform the sequential greedy locally for the APs in each CPU.
                - 'Q_random': select Q APs randomly for each CPU.
                - 'succesive_local_ES': perform the local exhaustive search for the APs in each CPU in a successive way, 
                i.e., when evaluating each CPU we use the information from previous CPUs.
                - 'succesive_local_SG': perform the local sequential greedy for the APs in each CPU in a successive way,    
                i.e., when evaluating each CPU we use the information from previous CPUs.
            # Improved the efficiency of the exhaustive search method, by considering only the AP assignments that use 
            the maximum number of APs in each CPU.

    # 20241219:
        # Added:
            # Update the function AP_OnOff_GlobalHeuristics() with the following heuristics:
                - 'bestgains_individualAPs': select the best AP according the their individual average channel gains 
                (average over all the users).


    # To do:
        # Adapt the system model for the AP on/off optimization problem (Done).
        # Implement the sequential greedy AP on/off (Done).
        # Study the independece of the AP on/off. Is sequential always optimal? -> NO (Done).




 

## Getting Started

Download links:

SSH clone URL: ssh://git@git.jetbrains.space/gtec/drl-sch/Cell-Free.git

HTTPS clone URL: https://git.jetbrains.space/gtec/drl-sch/Cell-Free.git



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

What things you need to install the software and how to install them.

```
Examples
```

## Deployment

Add additional notes about how to deploy this on a production system.

## Resources

Add links to external resources for this project, such as CI server, bug tracker, etc.
