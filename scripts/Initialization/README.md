# BlueBrain Pre-processing pipeline


## Dissection images

* **Whole hemisphere photos**: files follow this naming convention: 

    _CaseNumber_SliceNumber[Side]_
 
    and are placed under a directory named "Whole Slice". For instance: 

  * Whole Slice/P57-16_A2[P].jpg for the posterior surface of a cerebrum slice
  * Whole Slice/P57-16_A2[A].jpg for the anterior surface
  * Whole Slice/P57-16_C3[M].jpg for the medial surface of a cerebellar slice
  * Whole Slice/P57-16_C3[L].jpg for the lateral surface
  * Whole Slice/P57-16_B1[R].jpg for the rostral surface of a brainstem slice
  

* **Whole slice photos**: files follow this naming convention:
  
  _CaseNumber_SliceNumber[Side]_ 
  
  and are placed under a directory named "Whole Slice". For instance: 

  * Whole Slice/P57-16_A2[P].jpg for the posterior surface of a cerebrum slice
  * Whole Slice/P57-16_A2[A].jpg for the anterior surface
  * Whole Slice/P57-16_C3[M].jpg for the medial surface of a cerebellar slice
  * Whole Slice/P57-16_C3[L].jpg for the lateral surface
  * Whole Slice/P57-16_B1[R].jpg for the rostral surface of a brainstem slice


* **Photos of blocks**: files follow this naming convention:

  _CaseNumber_Blocks_SliceNumber[Side]_

  and are placed under a directory named "Blocks". For instance:
  * Blocks/P57-16_Blocks_A1[A].jpg
  
  If a slice only has one block, there will be no specific photo of the blocks 


* **Blockface volumes**: files follow this naming convention:

  _CaseNumber_Blocks_BlockNumber_

  and are placed under a directory named "BlockFacePhotoBlocks". For instance:
  * Blocks/P57-16_Blocks_A1.1.nii.gz
  
  Each block contains the (corrected) stack of RGB blockface photographs.

## Manual/semi-automatic processing: 
The initial manual/semi-automatic processing has already been run and stored in this repository.
It consists of pixel size correction in the "Whole slice photos" and "Photos of blocks", block detection
in "Photos of blocks" and registration between block and slice 2D photos.



## 3D registration (reconstruction) of blocks

The first step is running _RegisterCutfaceToBlockfaceAllRegions.m_.
This script registers the face of the block to the top/bottom of the 
corresponding blockface photo stack. It typically requires a fair amount of 
manual interaction. The script has a big outter loop for cerebrum, cerebellum 
and brainstem, each with its orientaton and peculiarity for the last slice:

* **Cerebrum blocks**. Note that the top of a block (i.e., first images in
the volume) correspond to the posterior side of the block, except for the
most posterior block, which is processed face down (on the flat side)
and is the other way around.


* **Cerebellum blocks**. Note that the top of a block (i.e., first images in
the volume) correspond to the lateral side, except for the most lateral
slice, which is the other way around, so the flat side can be down


* **Brainstem block**. Note that the top of a block (i.e., first images in
the volume) correspond to the superior (rostral) side. There are no
exceptions with the final block in the brainstem.

This script receives as input the "BlockVols" directory of the case, and 
creates a new directory BlockVols/registered with 3 files per block:

   BlockVols/registered/case_block_[target.png / warped.png / .mat]

For instance

* BlockVols/registered/P57-16_P5.1.target.png
* BlockVols/registered/P57-16_P5.1.warped.png
* BlockVols/registered/P57-16_P5.1.mat

The second step is to initialize the position of the blocks in 3D, independently for 
cerebrum, cerebellum, and  brainstem. You do this with the scripts: 

* _InitializeWithCutPhotosCerebrum.m_
* _InitializeWithCutPhotosCerebellum.m_
* _InitializeWithCutPhotosBrainstem.m_

They require a bit of interaction if the registrations of the (top-of-block) 
blockface to the mosaic of the previous slice are not satisfactory.

The scripts will ask for a "parent" output directory; it will automatically 
append "initializedBlocks" to your choice. Please choose the same in all 
three scripts; files will not be overwritten. 

These scripts produce two things: 

a) a bunch of initialized blocks, e.g., 

      initializedBlocks/P57-16_A3.2_volume.[gray/rgb/mask].initialized.mgz

b) three sets of mosaics with the whole cerebrum/cerebellum/brainstem

      initializedBlocks/initialResampledCerebrum.[gray/rgb/mask].nii.gz

      initializedBlocks/initialResampledCerebellum.[gray/rgb/mask].nii.gz

      initializedBlocks/initialResampledBrainstem.[gray/rgb/mask].nii.gz

Now you need a bit of manual intervention: we'll need 3 transforms in 
FreeSurfer lta format that align the mosaics with the ex vivo MRI scan.
I normally use averageWithReg.reoriented.nii.gz, which is a rotated version 
of the (averaged) original acquisition, so that it is properly oriented (note
that this resampling happened with the header). So, for each of the 3 mosaics
(cerebrum, cerebellum, brainstem):

a) Open the MRI in Freeview, with the mosaic on top (grayscale version because)
   Freeview doesn't handle transparency with RGB too well. For instance:

     freeview averageWithReg.reoriented.nii.gz  \
              initializedBlocks/initialResampledBrainstem.gray.nii.gz  

b) Use the tools under Tools -> Transform Volume  and the transparency settings
   to align the mosaic to the MRI. 

c) When you are done, click on "Save Reg" and save the registration as:
   initializedBlocks/initialResampled[Cerebrum/Cerebellum/Brainstem].regToMRI.lta

And now, you're ready to run the "monster"! This is done with the script 
"mosaicPuzzle/RefineWithConstrainedJointRegistration.m". As input you provide:

- the directory with initialized blocks, i.e., "initializedBlocks"

- the reoriented MRI scan, which I normally name "averageWithReg.reoriented.nii.gz"

The output directory should be called "registeredBlocks". Depending on the
resolution, this may run for many hours (on my laptop, it takes about a whole 
day at 0.5 mm resolution). 

Finally, the last steps consists of registering the histology sections to the blockface to initialize
the linear registration algorithm. This is done by running the _mosaicPuzzleUpdated/createDownsampleGapBlocks.m_
file.







