#!/usr/bin/env bash

# Linear algorithm
python ../LinearReconstruction/1.train_BT.py --subject P58-16
python ../LinearReconstruction/2.predict_BT.py --subject P58-16
python ../LinearReconstruction/3.generate_block_masks_easy.py --subject P58-16
python ../LinearReconstruction/4.generate_virtual_MRI_blocks.py --subject P58-16
python ../LinearReconstruction/5.generate_slices.py --subject P58-16



# ST algorithm
python ../NonLinearReconstruction/algorithm/1.initialize_graph.py --subject P58-16
python ../NonLinearReconstruction/algorithm/2.solve_st.py --nc 3 --c1 LFB --c2 HE --cost l1 --nn 2 --subject P58-16
python ../NonLinearReconstruction/algorithm/3.group_flow.py --subject P58-16
python ../NonLinearReconstruction/algorithm/3.1.group_flow_reverse.py --subject P58-16
python ../Reconstruction/3d-recon-labels.py --subject P58-16
python ../NonLinearReconstruction/algorithm/5.propagate_aparc.py --subject P58-16 --reg_algorithm ST3_L1_RegNet_NN2
python ../NonLinearReconstruction/algorithm/6.merge_cortical_labels.py --subject P58-16 --reg_algorithm ST3_L1_RegNet_NN2

# 3D reconstruction
python ../Reconstruction/3d-recon-histo.py --subject P58-16
python ../Reconstruction/3d-recon-labels.py --subject P58-16 --final_labels
python ../Reconstruction/3d-recon-all-labels.py --subject P58-16 --nn 2 --res 0.2
python ../Reconstruction/3d-recon-all.py --subject P58-16 --nn 2
python ../Reconstruction/3d-recon-histo-hr.py --subject P58-16 --reg_algorithm ST3_L1_RegNet_NN2
python ../Reconstruction/3d-recon-labels-hr.py --subject P58-16 --reg_algorithm ST3_L1_RegNet_NN2 --final_labels

# Webpage
python ../../webpage/create_material.py --subject P58-16 --nn 2 --do_histo_hr --do_histo

