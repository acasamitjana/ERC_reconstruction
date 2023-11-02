% this script initializes the reconstruction of the blockface stacks using
% the registrations between the cut photos and the blockface images at the
% top of the blocks. It requires a bit of interaction if the registrations
% of the (top-of-block) blockface to the mosaic of the previous slice are
% not satisfactory
% This script is for the cerebrum
clear all
clc

disp('Please select directory with autopsy photos of whole slices (typically: ''XXXX/Whole Slices''):')
wholeSliceDir = uigetdir('','Please select directory with autopsy photos of whole slices (typically: ''XXXX/Whole Slices''):');
wholeSliceDir(end+1)=filesep;

disp('Please select directory with autopsy photos of blocked slices (typically: ''XXXX/Blocks''):')
cutFileDir = uigetdir('','Please select directory with autopsy photos of blocked slices (typically: ''XXXX/Blocks''):');
cutFileDir(end+1)=filesep;

disp('Please select directory with blockface volumes (typically: ''XXXX/BlockVols''):')
blockFaceStackDir = uigetdir('','Please select directory with blockface volumes (typically: ''XXXX/BlockVols''):');
blockFaceStackDir(end+1)=filesep;

disp('Please select parent output directory (we will append "initializedBlocks" to your choice):')
outputdir = uigetdir('','Please select parent output directory (we will append "initializedBlocks" to your choice):');
outputdir=[outputdir filesep 'initializedBlocks' filesep];
 
%%%%%%%%%%%%%%%%

addpath(genpath(['.' filesep 'functions' filesep]));

% Child directories of interest
sliceMaskDir=[wholeSliceDir filesep 'cornersAndMasks' filesep];
sliceRegDir=[wholeSliceDir filesep 'registrations' filesep]; 
jigsawRegistrationDir=[cutFileDir filesep 'registrations' filesep];
blockFaceRegistrationDir=[blockFaceStackDir filesep 'registrations' filesep];

% Constants
RES_C=0.2;
RES_RESAMPLED=0.4; % to match the MRI
SECTION_THICKNESS=0.025;

% Create output dir if necessary
if exist(outputdir,'dir')==0
    mkdir(outputdir);
end

% Get number of (whole slices) and build vector of slices to work on
% We use the following convention: negative for P slices, positive for A
% slices.
d=dir([blockFaceStackDir filesep '*.gray.nii.gz']);
f=find(d(1).name=='_'); f2=find(d(1).name=='.');
casename=d(1).name(1:f(1)-1);
d=dir([blockFaceStackDir filesep casename '_A*.gray.nii.gz']);
f=find(d(end).name=='.');
NA=str2double(d(end).name(f(1)-1));
d=dir([blockFaceStackDir filesep casename '_P*.gray.nii.gz']);
f=find(d(end).name=='.');
NP=str2double(d(end).name(f(1)-1));
slices=[-NP:-1 1:NA]; 
nSlices=length(slices);

% This vector stores the thicknesses of the slices
thicknesses=zeros(1,nSlices);

% These two vectors stores the min/max RAS coordinates of the volumes,
% which we will use when resampling all the photographs into a single
% volume.
cornersRAScuboid1=Inf*ones(4,1);
cornersRAScuboid2=-Inf*ones(4,1);
maxSiz=0; % in X-Y plane, in voxels

% Main loop around slices
for s=1:nSlices 

    slice=abs(slices(s));
    if slices(s)<0
        direction='P';
    else
        direction='A';
    end
    
    if exist([outputdir filesep casename '_' direction num2str(abs(slices(s))) '.1_volume.rgb.initialized.1mgz'])
        disp(['Skipping  slice ' direction num2str(slice) ' (' num2str(s) ' of ' num2str(nSlices) ')']);
        d=dir([blockFaceStackDir filesep casename '_' direction num2str(slice) '.*.gray.nii.gz']);
        nBlock=length(d);
        BmriWarped=cell(1,nBlock);
        BmriMask=cell(1,nBlock);
        for Bind=1:nBlock
            filenameGray=[outputdir filesep casename '_' direction num2str(abs(slices(s))) '.' num2str(Bind) '_volume.gray.initialized.mgz'];
            BmriWarped{Bind}=myMRIread(filenameGray);
            BmriWarpedMask{Bind}=myMRIread([blockFaceStackDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '_volume.mask.nii.gz']);
            BmriWarpedMask{Bind}.vox2ras0 = BmriWarped{Bind}.vox2ras0;
            BmriWarpedMask{Bind}.volres = BmriWarped{Bind}.volres;
            filenameMask=[outputdir filesep casename '_' direction num2str(abs(slices(s))) '.' num2str(Bind) '_volume.mask.initialized.mgz'];
            BmriMask{Bind}=myMRIread(filenameMask);
        end
        
        load([blockFaceRegistrationDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.mat'],'shrinkingFactor');
        effectiveSectionThickness=SECTION_THICKNESS/shrinkingFactor;
        thicknessMap=sum(BmriMask{Bind}.vol/255,3)*effectiveSectionThickness;
        aux=thicknessMap(thicknessMap>2.5);
        % thicknessMeasured=median(aux);
        thicknessMeasured=prctile(aux,75);
        if slices(s)~=-NP && slices(s)~=NA
            if thicknessMeasured>3 && thicknessMeasured<7
                thickness=5;
            elseif thicknessMeasured>7 && thicknessMeasured<13
                thickness=10;
            elseif thicknessMeasured>13 && thicknessMeasured<17
                thickness=15;
            else
                disp(['Warning: estimated thickness is outside [3,7] and [8,12] for slice ' num2str(s) ', block ' num2str(Bind) ]);
                disp('Type ''return'' to continue...')
                keyboard
            end
        else
            thickness=thicknessMeasured;
        end
            
    else
        disp(['Working on slice ' direction num2str(slice) ' (' num2str(s) ' of ' num2str(nSlices) ')']);

        % Go around blocks, computing the preliminary alignment
        d=dir([blockFaceStackDir filesep casename '_' direction num2str(slice) '.*.gray.nii.gz']);
        nBlock=length(d);

        Bmri=cell(1,nBlock);
        BmriMask=cell(1,nBlock);
        BmriRGB=cell(1,nBlock);
        BmriWarped=cell(1,nBlock);
        BmriWarpedMask=cell(1,nBlock);

        for Bind=1:nBlock
            tic;
            % Read in block and corresponding mask
            Bmri{Bind}=myMRIread([blockFaceStackDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '_volume.gray.nii.gz']);
            BmriMask{Bind}=myMRIread([blockFaceStackDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '_volume.mask.nii.gz']);
            BmriRGB{Bind}=myMRIread([blockFaceStackDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '_volume.rgb.nii.gz']);

            % Compute thickness (force it to be 5 or 10, except for first and
            % last slice)
            load([blockFaceRegistrationDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.mat'],'shrinkingFactor');
            effectiveSectionThickness=SECTION_THICKNESS/shrinkingFactor;
            thicknessMap=sum(BmriMask{Bind}.vol/255,3)*effectiveSectionThickness;
            aux=thicknessMap(thicknessMap>2.5);
            % thicknessMeasured=median(aux);
            thicknessMeasured=prctile(aux,75);
            if slices(s)~=-NP && slices(s)~=NA
                if thicknessMeasured>3 && thicknessMeasured<7
                    thickness=5;
                elseif thicknessMeasured>8 && thicknessMeasured<15
                    thickness=10;
                else
                    disp(['Warning: estimated thickness is outside [3,7] and [8,12] for slice ' num2str(s) ', block ' num2str(Bind) ]);
                    disp('Type ''return'' to continue...')
                    keyboard
                end
            else
                thickness=thicknessMeasured;
            end

            % Make sure thickness is constant across blocks of the same
            % slice...
            if Bind==1
                thicknesses(s)=thickness;
            else
                if thicknesses(s)~=thickness
                    disp(['Warning: thickness is not constants for blocks in slice ' num2str(s) ]);
                    disp('Type ''dbcont'' to continue...')
                    keyboard
                end
            end

            % Compute vox2ras by concatenating transforms (requires registration to
            % mosaic)
            % First, block-face to cut photograh of blocks
            load([blockFaceRegistrationDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.mat'],'tform');
            tt=tform.T';
            T1=eye(4); T1(1:2,1:2)=tt(1:2,1:2); T1(1:2,4)=tt(1:2,3);
            % Second, cut photograph of blocks to cut photo of whole slice
            if nBlock>1
                load([jigsawRegistrationDir filesep casename '_' direction num2str(slice) '[P].transforms.mat'],'tforms');
                tt=tforms{Bind}.T';
            else
                tt=eye(3);
            end
            T2=eye(4); T2(1:2,1:2)=tt(1:2,1:2); T2(1:2,4)=tt(1:2,3);
            Ttot=T2/T1; %T2*inv(T1); % this is the transform we've got so far


            % Dfine vox2ras0, and incorporate translation in z (anterior-posterior;
            % in the future we could rotate the header at the end so z is inferior-superior as
            % it should, rather than posterior anterior, as it is now...)
            vox2ras0=diag([RES_C RES_C effectiveSectionThickness 1])*Ttot;
            volres=[norm(vox2ras0(1:2,1))  norm(vox2ras0(1:2,2)) effectiveSectionThickness];
            if s==1
                vox2ras0(3,4)=-size(BmriMask{Bind}.vol,3)*effectiveSectionThickness + thicknesses(1);

            else
                vox2ras0(3,4)=sum(thicknesses(1:s-1));
            end

            % This is the one I'll write to disk; for some reason, vox2ras/volres
            % are messed up when I use the windows compatible writer... go figure
            % It doesn't really matter because I redefined the vox2ras0 anyway...
            BmriWarped{Bind}=[];
            BmriWarped{Bind}.vox2ras0=vox2ras0;
            BmriWarped{Bind}.vol=Bmri{Bind}.vol;
            BmriWarped{Bind}.volres=volres;
            BmriWarpedMask{Bind}=BmriWarped{Bind};
            BmriWarpedMask{Bind}.vol=BmriMask{Bind}.vol;
            toc
            maxSiz=max([maxSiz size(BmriWarped{Bind}.vol,1) size(BmriWarped{Bind}.vol,2)]);
        end



            % Write blocks to disk
        for Bind=1:nBlock  

            mri=BmriWarped{Bind};

            filenameGray=[outputdir filesep casename '_' direction num2str(abs(slices(s))) '.' num2str(Bind) '_volume.gray.initialized.mgz'];
            myMRIwrite(mri,filenameGray);

            filenameMask=[outputdir filesep casename '_' direction num2str(abs(slices(s))) '.' num2str(Bind) '_volume.mask.initialized.mgz'];
            mri.vol=BmriMask{Bind}.vol;
            myMRIwrite(mri,filenameMask);

            filenameRGB=[outputdir filesep casename '_' direction num2str(abs(slices(s))) '.' num2str(Bind) '_volume.rgb.initialized.mgz'];
            mri.vol=BmriRGB{Bind}.vol;
            myMRIwrite(mri,filenameRGB);

        end
        clear BmriMask BmriRGB
        
        % Make (preliminary) mosaics of this slice
        if s==1, zshift=0.8*thickness; else, zshift=BmriWarped{1}.vox2ras0(3,4)+0.8*thickness; end
        mriMosaicA=makeMosaic(BmriWarped,BmriWarpedMask,zshift,RES_C,effectiveSectionThickness);
        if s==1, zshift=0.05*thickness; else, zshift=BmriWarped{1}.vox2ras0(3,4)+0.05*thickness; end
        mriMosaicP=makeMosaic(BmriWarped,BmriWarpedMask,zshift,RES_C,effectiveSectionThickness);

        % Register the mosaics and propagate transforms to blocks (rewrite)
        if s==1 % First slice is most posterior
            mriMosaicAprev=mriMosaicA;
            mriMosaicPprev=mriMosaicP;
        else
            tform = registerTillHappy(mriMosaicAprev.vol,mriMosaicP.vol);
            tt=tform.T';
            T3=eye(4); T3(1:2,1:2)=tt(1:2,1:2); T3(1:2,4)=tt(1:2,3);
            % Tras=mriMosaicAprev.vox2ras0*T3*inv(mriMosaicP.vox2ras0);
            Tras=(mriMosaicAprev.vox2ras0*T3)/mriMosaicP.vox2ras0;
            Tras(3,3)=1; Tras(3,4)=0; % we don't want anything in z
            for Bind=1:nBlock
                BmriWarped{Bind}.vox2ras0=Tras*BmriWarped{Bind}.vox2ras0;
            end
            mriMosaicAprev=mriMosaicA;
            mriMosaicAprev.vox2ras0=Tras*mriMosaicAprev.vox2ras0;
            mriMosaicPprev=mriMosaicP;
            mriMosaicPprev.vox2ras0=Tras*mriMosaicPprev.vox2ras0;
        end
        
    end
    
    
    
    % we also gather the absolute corners (useful for resampling
    % later on)
    for Bind=1:nBlock
        for i=[0 maxSiz-1]
            for j=[0 maxSiz-1]
                for k=[0 size(BmriWarped{Bind}.vol,3)]
                    aux=BmriWarped{Bind}.vox2ras0*[i; j; k; 1];
                    cornersRAScuboid1=min(cornersRAScuboid1,aux);
                    cornersRAScuboid2=max(cornersRAScuboid2,aux);
                end
            end
        end
    end
    

end


% OK now we do the resampling to build a single volume,with whatever
% resolution we want (we can use 0.4 to match MRI)
disp('Building single resampled volume');

% Define header and size
v2r0=[RES_RESAMPLED 0 0 cornersRAScuboid1(1);
    0 RES_RESAMPLED 0 cornersRAScuboid1(2);
    0 0 RES_RESAMPLED cornersRAScuboid1(3);
    0 0 0 1];
vr=[RES_RESAMPLED RES_RESAMPLED RES_RESAMPLED];  
siz=ceil((cornersRAScuboid2(1:3)-cornersRAScuboid1(1:3))/RES_RESAMPLED);
mriResampled=[];
mriResampled.vol=zeros(siz');
mriResampled.vox2ras0=v2r0;
mriResampled.volres=vr;

% Now go around volumes and resample
numerator=zeros(size(mriResampled.vol));
numeratorRGB=zeros([size(mriResampled.vol) 3]);
denominator=eps+zeros(size(mriResampled.vol));

[gr1,gr2,gr3]=ndgrid(1:size(mriResampled.vol,1),1:size(mriResampled.vol,2),1:size(mriResampled.vol,3));
voxResampled=[gr2(:)'; gr1(:)'; gr3(:)'; ones(1,numel(gr1))];
rasResampled=mriResampled.vox2ras0*voxResampled;
d1=dir([outputdir filesep casename '_A*_volume.gray.initialized.mgz']);
d2=dir([outputdir filesep casename '_P*_volume.gray.initialized.mgz']);
d=[d1 ; d2];
for i=1:length(d)
    disp(['  Volume ' num2str(i) ' of ' num2str(length(d))]);
    mri=myMRIread([outputdir filesep d(i).name]);
    mriMask=myMRIread([outputdir filesep d(i).name(1:end-21) '.mask.initialized.mgz']);
    mriRGB=myMRIread([outputdir filesep d(i).name(1:end-21) '.rgb.initialized.mgz']);
    
    % voxB=inv(mri.vox2ras0)*rasResampled;
    voxB=mri.vox2ras0\rasResampled;
    v1=voxB(2,:);
    v2=voxB(1,:);
    v3=voxB(3,:);
    ok=v1>1 & v2>1 & v3>1 & v1<=size(mri.vol,1) & v2<=size(mri.vol,2) & v3<=size(mri.vol,3);
    vals=zeros(1,size(rasResampled,2));
    
    vals(ok)=interp3(mri.vol,v2(ok),v1(ok),v3(ok));
    intensities=zeros(size(mriResampled.vol));
    intensities(:)=vals;
    
    vals(ok)=interp3(mriRGB.vol(:,:,:,1),v2(ok),v1(ok),v3(ok));
    intensitiesR=zeros(size(mriResampled.vol));
    intensitiesR(:)=vals;
    vals(ok)=interp3(mriRGB.vol(:,:,:,2),v2(ok),v1(ok),v3(ok));
    intensitiesG=zeros(size(mriResampled.vol));
    intensitiesG(:)=vals;
    vals(ok)=interp3(mriRGB.vol(:,:,:,3),v2(ok),v1(ok),v3(ok));
    intensitiesB=zeros(size(mriResampled.vol));
    intensitiesB(:)=vals;
        
    vals(ok)=interp3(mriMask.vol,v2(ok),v1(ok),v3(ok));
    mask=zeros(size(mriResampled.vol));
    mask(:)=double(vals)/255;
    
    numerator = numerator + mask.*intensities;
    aux=numeratorRGB(:,:,:,1); aux=aux+mask.*intensitiesR; numeratorRGB(:,:,:,1)=aux;
    aux=numeratorRGB(:,:,:,2); aux=aux+mask.*intensitiesG; numeratorRGB(:,:,:,2)=aux;
    aux=numeratorRGB(:,:,:,3); aux=aux+mask.*intensitiesB; numeratorRGB(:,:,:,3)=aux;
    denominator=denominator+mask;
end

mriResampled.vol=numerator./denominator;
myMRIwrite(mriResampled,[outputdir filesep 'initialResampledCerebrum.gray.nii.gz']);
mriResampled.vol=numeratorRGB./repmat(denominator,[1 1 1 3]);
myMRIwrite(mriResampled,[outputdir filesep 'initialResampledCerebrum.rgb.nii.gz']);
mriResampled.vol=255*double(denominator>0.5); 
myMRIwrite(mriResampled,[outputdir filesep 'initialResampledCerebrum.mask.nii.gz']);

disp('All done!');






