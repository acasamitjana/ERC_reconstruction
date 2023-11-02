clear all
clc
tic

YamlStruct = ReadYaml([pwd() filesep '..' filesep 'configFile_P58-16.yaml']);

% Directory with registered blocks, typically initializedBlocks
disp('Please select the input directory (i.e. "initializedBlocks")')
initialDir = uigetdir('','Please select the input directory (i.e. "initializedBlocks")');


% This is a directory with the downsampled blocks.
disp('Please select parent output directory (we will append "initHistoBlocksDownsampled1mm_GAP" to your choice):')
downsampledDir = uigetdir('','Please select parent output directory (we will append "initHistoBlocksDownsampled1mm_GAP" to your choice):');
downsampledDir=[outputdir filesep 'initHistoBlocksDownsampled1mm_GAP' filesep];

% Input MRI
disp('Please select the MRI directory containing mri volume and masks')
mriDir = uigetdir('','Please select the MRI directory containing mri volume and masks:');
inputMRI = [mriDir filesep 'mri.nii.gz']
MRI_SEG_OR_THRESHOLD=[mriDir filesep 'mask_mri.nii.gz']
MRI_SEG_OR_THRESHOLD_DILATED=[mriDir filesep 'mask_mri.dilated.nii.gz']
MRI_SEG_OR_THRESHOLD_CLL=[mriDir filesep 'mask_mri.cerebellum.nii.gz']
MRI_SEG_OR_THRESHOLD_CR=[mriDir filesep 'mask_mri.cerebrum.nii.gz']
MRI_SEG_OR_THRESHOLD_BS=[mriDir filesep 'mask_mri.brainstem.nii.gz']

% Voxel size at which we work
downsampledVoxSize=[1 1 0.25]; % downsampledVoxSize=1.0; % for development (faster)

% You can either provide a binary segmentation mask or compute it by thresholding

addpath(genpath(['..' filesep 'functions' filesep]));
addpath(genpath(['.' filesep 'L-BFGS-B-C' filesep]));
%addpath(genpath(['.' filesep 'samsegMRI.m']));
%addpath(genpath(YamlStruct.freesurfer6));


%%%%%%%%%%%%  END OF OPTIONS %%%%%%%%%%%%%%%%%
% Some constants...
CEREBRUM=1;
CEREBELLUM=2;
BRAINSTEM=3;

if exist(downsampledDir,'dir')==0
    mkdir(downsampledDir);
end


% Downsample blocks: MRI and blocks (gray/rgb/mask)
% MRI scan
MRIdownsampled=[downsampledDir filesep 'DownsampledMRI.mgz'];
MRIdownsampledMask=[downsampledDir filesep 'DownsampledMRI.mask.mgz'];
MRIdownsampledMaskCLL=[downsampledDir filesep 'DownsampledMRI.mask.cerebellum.mgz'];
MRIdownsampledMaskCR=[downsampledDir filesep 'DownsampledMRI.mask.cerebrum.mgz'];
MRIdownsampledMaskBS=[downsampledDir filesep 'DownsampledMRI.mask.bs.mgz'];
MRIdownsampledMaskDilated=[downsampledDir filesep 'DownsampledMRI.mask.dilated.mgz'];
suffix='initialized';
prefix='LFB';
LS=length(suffix);
LP=length(prefix);

disp('  Downsampling MRI scan');
mri=MRIread(inputMRI);
factor=downsampledVoxSize./mri.volres;
mri2 = downsampleMRI(mri,factor);
mri2.vol=mri2.vol/max(mri2.vol(:));
myMRIwrite(mri2,MRIdownsampled);

disp('  Preparing MRI CLL mask');
aux=MRIread(MRI_SEG_OR_THRESHOLD_CLL);
aux.vol=double(aux.vol>0);
factor=downsampledVoxSize./aux.volres;
aux2 = downsampleMRI(aux,factor);
MRI_MASK=aux2.vol>0.5;
aux2.vol=255*double(MRI_MASK);
myMRIwrite(aux2,MRIdownsampledMaskCLL,'float');
        
disp('  Preparing MRI CR mask');
aux=MRIread(MRI_SEG_OR_THRESHOLD_CR);
aux.vol=double(aux.vol>0);
factor=downsampledVoxSize./aux.volres;
aux2 = downsampleMRI(aux,factor);
MRI_MASK=aux2.vol>0.5;
aux2.vol=255*double(MRI_MASK);
myMRIwrite(aux2,MRIdownsampledMaskCR,'float');

disp('  Preparing MRI BS mask');
aux=MRIread(MRI_SEG_OR_THRESHOLD_BS);
aux.vol=double(aux.vol>0);
factor=downsampledVoxSize./aux.volres;
aux2 = downsampleMRI(aux,factor);
MRI_MASK=aux2.vol>0.5;
aux2.vol=255*double(MRI_MASK);
myMRIwrite(aux2,MRIdownsampledMaskBS,'float');

disp('  Preparing MRI mask');
aux=MRIread(MRI_SEG_OR_THRESHOLD);
aux.vol=double(aux.vol>0);
factor=downsampledVoxSize./aux.volres;
aux2 = downsampleMRI(aux,factor);
MRI_MASK=aux2.vol>0.5;
aux2.vol=255*double(MRI_MASK);
myMRIwrite(aux2,MRIdownsampledMask,'float');

disp('   Preparing MRI mask dilated');
aux=MRIread(MRI_SEG_OR_THRESHOLD_DILATED);
aux.vol=double(aux.vol>0);
factor=downsampledVoxSize./aux.volres;
aux2 = downsampleMRI(aux,factor);
MRI_MASK=aux2.vol>0.5;
aux2.vol=255*double(MRI_MASK);
myMRIwrite(aux2,MRIdownsampledMaskDilated,'float');

% Read in manual transforms, if available
LTAcerebellum=my_lta_read([initialDir filesep 'initialResampled.Cerebellum.regToMRI.ras2ras.lta']);
LTAbrainstem=my_lta_read([initialDir filesep 'initialResampled.Brainstem.regToMRI.ras2ras.lta']);
LTAcerebrum=my_lta_read([initialDir filesep 'initialResampled.Cerebrum.regToMRI.ras2ras.lta']);

% Block downsampling
disp('Downsampling blocks');
d=dir([initialDir filesep '*mask.' suffix '.mgz']); % Need to be changed to: 
Nblocks=length(d);

for i=29:29%Nblocks
    
    disp(['  Block ' num2str(i) ' of ' num2str(Nblocks) ': ' d(i).name]);
%       continue
    % Find corresponding initial registration and slice indices
    if num2str(d(i).name(end-20-LS-LP-1))=='C' % cerebellum   % DAdria: make sure this works with the current naming convention
        LTA=LTAcerebellum;
    elseif num2str(d(i).name(end-20-LS-LP-1))=='B' % brainstem   % DAdria: make sure this works with the current naming convention
        LTA=LTAbrainstem;
    else
        LTA=LTAcerebrum;
    end
    
    % Block Mask
    input=[initialDir filesep d(i).name];
    output=[downsampledDir filesep d(i).name(1:end-5-LS) '.downsampled.mgz'];   % DAdria: please fix this naming convention if needed
    mri = MRIread(input);
%     mri.vol=permute(mri.vol,[2 1 3]);
    factor=downsampledVoxSize./mri.volres;
    mri2 = downsampleMRI_GAP(mri,factor(1:2));
    mri2.vox2ras0 = LTA * mri2.vox2ras0;
    myMRIwrite(mri2,output);

    % Block Gray
    input=[initialDir filesep d(i).name(1:end-9-LS) 'gray.' suffix '.mgz'];     % DAdria: please make sure this works with the usual naming convention
    output=[downsampledDir filesep d(i).name(1:end-9-LS) 'gray.downsampled.mgz'];
    mri = MRIread(input);
%     mri.vol=permute(mri.vol,[2 1 3]);
    factor=downsampledVoxSize./mri.volres;
    mri2 = downsampleMRI_GAP(mri,factor(1:2));
    mri2.vox2ras0 = LTA * mri2.vox2ras0;
    myMRIwrite(mri2,output);

end


