clc
clear all

FREESURFER_HOME = '/home/acasamitjana/Software_MI/freesurfer_exvivo';
FS_PYTHON_DEV = '/home/acasamitjana/Software_MI/freesurfer_exvivo/python/scripts';
codedir = '/home/acasamitjana/Repositories/BUNGEE-TOOLS-Pipeline/mosaicPuzzleUpdated';
results_dir = '/home/acasamitjana/Data/P41-16/P41-16_scanned_20180124/samseg';
MRI_mgz_version = '/home/acasamitjana/Data/P41-16/P41-16_scanned_20180124/prova.mgz';

setenv('FREESURFER_HOME', FREESURFER_HOME);
setenv('FS_PYTHON_DEV', FS_PYTHON_DEV);
setenv('CODE_HOME', codedir);

cmd = ['sh ', codedir,'/run_seg ',...
        '-o ', results_dir,' -i ', MRI_mgz_version,' ',...
        '--threads 32'];

dlist_segmentations = dir(fullfile(results_dir,'*crispSegmentation.nii'));

if isempty(dlist_segmentations)
    try
        fprintf(['Segmenting case ',MRI_mgz_version,'\n'])
        [segstatus,segresult]=system(cmd);
        if segstatus
            warning(['problem with segmentation of ',MRI_mgz_version])
        end
    catch
        warning(['problem with segmentation of ',MRI_mgz_version])
    end
else
    fprintf(['Skipping case ',MRI_mgz_version,'\n'])
end
