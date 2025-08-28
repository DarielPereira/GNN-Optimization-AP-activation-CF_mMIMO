# GNN-based Optimization of sum-SE in Cell-Free Massive MIMO Communications with Limited Fronthaul Capacity

This project implements a framework for optimizing the activation of Access Points (APs) in cell-free massive MIMO 
systems with limited fronthaul capacity. The optimization is tested using heuristic methods and Graph Neural Networks (GNNs).

See details in the paper:
- **Title**: Pixel-Based CF-mMIMO: Addressing the AP Cooperation Cluster Formation in Fronthaul-Limited O-RAN Architectures
- **Authors**: Dariel Pereira Ruisánchez, Michael Joham, Óscar Fresnedo, Darian Pérez Adán, Luis Castedo, and Wolfgang Utschick
- **URL**: https://www.authorea.com/users/683174/articles/1322673-pixel-based-cf-mmimo-addressing-the-ap-cooperation-cluster-formation-in-fronthaul-limited-o-ran-architectures

---

## Project Structure

### Main Files and Scripts

- **`AP_OnOff_{}.py, {CDF,Grid,Ks,Ls_Ts}`**
  This files generate the results for the AP on/off optimization problem using different setups. The placeholders 
`{CDF,Grid,Ks,Ls_Ts}` indicate different configurations:

- **`AP_OnOff_SampleGeneration.py`**  
  Generates training samples for the GNN models. It creates datasets with graph-related information for the AP on/off task.

- **`AP_training.py`**  
  This script implements the training of a Graph Neural Network (GNN) to optimize the activation of Access Points (APs)

- **`functionsGraphHandling.py`**  
  This module contains classes and functions for handling graph-based data structures. It includes implementations of
Graph Neural Network (GNN) models.

- **`functionsSetup.py`**  
  Generates the system setup, including AP and UE positions, channel realizations, and other parameters.

- **`functionsAPAllocation.py`**  
  Implements heuristic methods for pilot allocation and AP on/off configuration.

- **`AP_OnOff_Heuristics.py`**  
  Calls the heuristic methods for AP on/off optimization.

- **`functionsUtils.py`**  
  Provides utility functions for loading results, saving data, and other general-purpose tasks.

- **`README.md`**  
  This file provides an overview of the project, its structure, and usage instructions.

---

## Libraries Used

The project relies on the following Python libraries:

- **`torch`**: For building and training the GNN models.
- **`torch_geometric`**: For handling graph data and implementing graph-based neural networks.
- **`numpy`**: For numerical computations.
- **`matplotlib`**: For plotting and visualizing results.
- **`tqdm`**: For progress bars during data generation and training.
- **`random`**: For generating random setups and seeds.

---


## Recommended use
-  Generate training samples using AP_OnOff_SampleGeneration.py.
-  Train the GNN models using AP_training.py.
-  Evaluate the models using the heuristic methods in AP_OnOff_Heuristics.py or AP_OnOff_{} scripts for 
different network configurations.

---

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

    # 20241220:
        # Added:
            # We include the script AP_OnOff_SampleGeneration.py to generate the samples for training the GNNs for 
            the AP on/off task. The training data is stored in the folder //AP_trainingData/newData. Data is moved to 
            folder .../inDataSet at the beginning of the script AP_training.
            # We include the script AP_training.py to train the GNNs for the AP on/off task. 
            # We modify the methods in the script functionsGraphHandling.py to handle the new data format for the AP 
            on/off task.

    # 20250122:
        # Added:
            # We update the function AP_OnOff_GlobalHeuristics() in the script functionsAllocation.py to include the 
            GNN-based prediction of the AP on/off.
            # We modify the AP_training.py script to simplify the load of the datasets.
            # We update the AP_OnOff_SampleGeneration.py script so datasets are directly generated by this script.
            # We update the functionsGraphHandling.py. We update all the clases and redesign the strcuture of the 
            neural networks (training in this setups achieves very low validation errors, that is, approx 0.06).
            # We modify the functionsSetup.py to allow the creation of non-square L AP distributions.
            # The AP_TrainingData folder includes the datasets and models evaluated in this stage.
            # The ResultsMediumSetups.xlsx file includes the results of the evaluation of the GNN models.

        # Some results: training performs very good, both training and validation data tend to low values of BCE loss. 
            However, although the trained model is competitive with the best heuristics, it is not able to outperform 
            them. The best guess is that the training samples fail to capture the actual complexity of the global 
            problem >> this problem holds and we are do not know yet how to solve it. Same problem happens to the
            GNN-gains model.
            
    # 20250128:
        # Added:
            # We update all the scripts to include the calls to the Gain-based GNN model.
            # We update the functionsGraphHandling.py script. Now, we have a parent class for the GNN models, and
            two child classes, one for the GNN model based on the correlation matrices and another for the GNN model 
            based on the average channel gains. 
            # In the folder AP_TrainingData, we have independent datasets and models for each GNN type. 

        # Some results: the GNN-gains achieves a competitive perfomance in both evaluation scenarios (T=4,Q=2 and T=8,Q=4).
        However, it achieves the best performance for the less trainined version of the model, that is, the model trained
        over only 7 epochs. Besides, the best performance is achieved when using f=1. Do not know how to justify.

    # 20250305:
        # Added:
            # The script AP_training_self_supervised.py was created and moved to the files of discarded scripts (__Unused).
            Self-supervised training cannot be implemented because the objective functions is not continuos or desrivable 
            with respecto the network parameters.


    # To do:
        # Adapt the system model for the AP on/off optimization problem (Done).
        # Implement the sequential greedy AP on/off (Done).
        # Study the independece of the AP on/off. Is sequential always optimal? -> NO (Done).
        # Test the training of the GNNs for the AP on/off task -> In process...
        # Modify the generation of the training samples to try to represent better the global networks -> In process...



