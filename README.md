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


    # To do:
        # Adapt the system model for the AP on/off optimization problem (Done).
        # Implement the sequential greedy AP on/off (Done).
        # Study the independece of the AP on/off. Is sequential always optimal?




 

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
