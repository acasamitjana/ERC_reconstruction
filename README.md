# Blue BRAIN reconstruction pipeline

This repository contains the scripts for 3D histology reconstruction.



<p align="left">
<img src="https://github.com/acasamitjana/ERC_reconstruction/blob/main/webpage/gif-EX9-19.gif?raw=true" alt="drawing" width="350"/>
</p>

## Data
You can download the project's data here: doi.org/10.5522/04/24243835. More details about the data acquisition can be found in [1]

## Code
The 3D reconstruction code is place in the _scripts_ directory and is organised as follows:
* **Initialisation**: Matlab code used to initialize the linear reconstruction via stack of blockface photographs
* **Linear step**: it contains several scripts to run the linear registration step between all blocks and the MRI. It is used to accomodate all blocks in the 3D space minimising both the overlap and separation between blocks. The final outcome is the matching MRI slice to each of the histology sections (LFB and H&E)
* **Non-linear step**: this non-linear step is used to refine the alignment between each MRI and histological slices (LFB and H&E) such that the overall reconstruction is accurate and smooth. More details about this step can be found in [2]
* **Reconstruction**: final reconstruction scripts that read the results from the previous steps and original data to compute the final reconstruction avoiding multiple resampling steps.
* **Pipeline**: scripts running the whole 3D reconstruction pipeline for each case

### Pre-processing
This Matlab code is used to initialize the linear reconstruction via stack of blockface photographs. Some of the manual work has already been done. This folder contains a README file with more instructions about the data used and the processing steps.
Main scripts:

1. _RegisterCutfaceToBlockfaceAllRegions.m_: registers the face of each block to the top/bottom blockface photo stack.
2. _InitializeWithCutPhotos*.m_: independent scripts for each structure (cerebrum, cerebellum and brainstem) that use information from previous steps to initialize the location of each block.
3. _createDownsampleGapBlocks.m_: registerts LFB sections to the corresponding blockface photo to generate initial LFB blocks.

### Linear reconstruction
The goal is to slice the MRI volume so that each LFB and H&E sections have their equivalent MRI slice. For that, we run a hieararchical linear registration algorithm to accomodate each LFB block in a single ``LFB volume'' that closely matches the MRI volume. Each MRI voxel is assigned to a given block by using a distance transform to deal with ambiguities

Main scripts:
1. _train_BT.py_: compute affine parameters for each block.
2. _predict_BT.py_: compute the aligned LFB blocks.
3. _generate_block_masks_easy.py_: assign an histology block to each MRI voxel.
4. _generate_virtual_MRI_blocks.py_: re-slice the MRI according to each block.
5. _generate_slices.py_: prepare MRI, LFB and H&E sections to for non-linear reconstruction. The different operations include resampling at a given resolution (typically 0.1mm) and independent linear alignment of each histology section to their MRI counterpart.

### Non-linear reconstruction
The remaining difference between each MRI, LFB and H&E section is non-linear. We run the ST3 algorithm presented in [2] to compute the final deformation field for each histology section.  The inverse deformation is used to propagate freesurfer parcellations to the manual delineations of the cortex on the LFB sections. If, for whatever reason, a block is not processed through the pipeline, we compute the registration from the linear alignment to the MRI.

1. _initialize_graph.py_: compute pairwise intermodal registration (MRI, LFB and H&E) and between neighbouring MRI slices.
2. _solve_st.py_: solve the ST3 algorithm given the pairwise registrations from step 1.
3. _group_flow.py_: compute the final deformation field for each histology section as the concatenation of affine and non-linear deformation fields.
4. _mosaic_registration_missing_block.py_: final registration of un-processed blocks (typically due to small number of sections or tissue, such that the most inferior brain-stem blocks)
5. _propagate_aparc.py_: propagate freesurfer parcellation from MRI to histology.
6. _merge_cortical_labels.py_: merge subocrtical labels with cortical parcellation.



## Bibliography
[1] Mancini, Matteo, et al. "A multimodal computational pipeline for 3D histology of the human brain." Scientific reports 10.1 (2020): 13839.

[2] Casamitjana, Adrià, et al. "Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas." Medical image analysis 75 (2022): 102265.
