# Blue BRAIN reconstruction pipeline

This repository contains the scripts for 3D histology reconstruction.

<p align="center">
<img src="https://github.com/acasamitjana/ERC_reconstruction/blob/main/webpage/init-pic.png?raw=true" alt="drawing" width="300"/>
</p>

## Data
You can download the project's data here: doi.org/10.5522/04/24243835. More details about the data acquisition can be found in [1]

## Code
The 3D reconstruction code is place in the _scripts_ directory and is organised as follows:
* **Initialisation**: Matlab code used to initialize the linear reconstruction via stack of blockface photographs
* **Linear step**: it contains several scripts to run the linear registration step between all blocks and the MRI. It is used to accomodate all blocks in the 3D space minimising both the overlap and separation between block. The final outcome is the matching MRI slice to each of the histology sections (LFB and H&E)
* **Non-linear step**: this non-linear step is used to refine the alignment between each MRI and histological slices (LFB and H&E) such that the overall reconstruction is accurate and smooth. More details about this step can be found in [2]
* **Reconstruction**: final reconstruction scripts that read the results from the previous steps and original data to compute the final reconstruction avoiding multiple resampling steps.
* **Pipeline**: scripts running the whole 3D reconstruction pipeline for each case

### Pre-processing
This Matlab code is used to initialize the linear reconstruction via stack of blockface photographs. Some of the manual work has already been done.

### Linear reconstruction
1. train_BT.py
2. predict_BT.py
3. generate_block_masks_easy.py
4. generate_virtual_MRI_blocks.py
5. generate_slices.py
6. check_linear_alignment.py

### Non-linear reconstruction

1. initialize_graph.py
2. solve_st.py
3. group_flow.py
4. mosaic_registration_missing_block.py
5. propagate_aparc.py
6. merge_cortical_labels.py



## Bibliography
[1] Mancini, Matteo, et al. "A multimodal computational pipeline for 3D histology of the human brain." Scientific reports 10.1 (2020): 13839.

[2] Casamitjana, Adrià, et al. "Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas." Medical image analysis 75 (2022): 102265.
