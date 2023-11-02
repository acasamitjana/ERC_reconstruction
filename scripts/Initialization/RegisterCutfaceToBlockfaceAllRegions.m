% registers the face of the block to the top/bottom of the corresponding
% blockface photo stack. It typically requires a fair amount of manual
% interaction.
% This script has a big outter loop for cerebrum, cerebellum and brainstem
% each with its orientaton and peculiarity for the last slice:
% - Cerebrum blocks. Note that the top of a block (i.e., first images in
% the volume) correspond to the posterior side of the block, except for the
% most posterior block, which is processed face down (on the flat side)
% and is the other way around.
% - Cerebellum blocks. Note that the top of a block (i.e., first images in
% the volume) correspond to the lateral side, except for the most lateral
% slice, which is the other way around, so the flat side can be down
% - Brainstem block. Note that the top of a block (i.e., first images in
% the volume) correspond to the superior (rostral) side. There are no
% exceptions with the final block in the brainstem.
clear
clc
close all

addpath(genpath(['.' filesep 'functions' filesep]));
addpath rigidTransform


% approximate shrinking in each dimension due to tissue processing
% We've got good empirical evidence that it's close to 0.80
APPROX_SHRINK_FACTOR=0.8;
% How many pictures to skip at the beginning of the block
SKIP=5;
% resolution of cut photos
RES_C=0.2;
% resolution of blockface images
RES_B=0.1;

% Input directories with block volumes, block cut photos & slice cut photos
disp('Please select directory with block volumes');
blockFaceStackDir = uigetdir('','Please select directory with block volumes:');
disp('Please select directory with block cut photos');
cutFileDir = uigetdir('','Please select directory with block cut photos:');
disp('Please select directory with slice cut photos');
wholeSliceDir = uigetdir('','Please select directory with slice cut photos:');

outputDir=[blockFaceStackDir filesep 'registrations' filesep];
if exist(outputDir,'dir')==0, mkdir(outputDir); end


for structure_index=3:3%1:3
    disp('**************');
    if structure_index==1
        disp('*  Cerebrum  *');
        ORI1='A';
        ORI2='P';
        d=dir([blockFaceStackDir filesep '*.rgb.nii.gz']);
        ok=zeros(1,length(d));
        for i=1:length(d)
            f=find(d(i).name=='.');
            if ~isempty(f) && f(1)>2 && ...
                    (d(i).name(f(1)-1)==ORI1 || d(i).name(f(1)-1)==ORI2 || ...
                    d(i).name(f(1)-2)==ORI1 || d(i).name(f(1)-2)==ORI2)
                ok(i)=1;
            end
        end
        d=d(ok>0);
        
    elseif structure_index==2
        disp('* Cerebellum *');
        ORI1='M';
        ORI2='L';
        d=dir([blockFaceStackDir filesep '*_C*.rgb.nii.gz']);
        
    else
        disp('* Brainstem  *');
        ORI1='C';
        ORI2='R';
        d=dir([blockFaceStackDir filesep '*_B*.rgb.nii.gz']);
        
    end
    disp('**************');

    for i=1:%length(d) 
        
        disp(['Working on case ' num2str(i) ' of ' num2str(length(d)) ': ' d(i).name]);
        
        blockFaceStackName=[blockFaceStackDir filesep d(i).name];
        blockFaceStackMaskName=[blockFaceStackDir filesep d(i).name(1:end-11) '.mask.nii.gz'];
        f=find(d(i).name=='_'); f2=find(d(i).name=='.');
        casename=d(i).name(1:f(1)-1);
        direction=d(i).name(f(1)+1);
        slice=str2double(d(i).name(f(1)+2:f2(1)-1));
        Bind=str2double(d(i).name(f2(1)+1));
        
        if exist([outputDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.1warped.png'])
            continue
        end
        
        % most posterior block in cerebrum / lateral in cerebellum:
        % we make an exception, and register to the anterior/medial face of the whole slice picture
        if i==length(d) && structure_index<=2
            slInd=slice-1;
        else
            slInd=slice;
        end    
        cutFileName=[cutFileDir filesep casename '_Blocks_' direction num2str(slInd) '[' ORI2 '].JPG'];
        cutMatFileName=[cutFileDir filesep 'cornersAndMasks' filesep casename '_Blocks_' direction num2str(slInd) '[' ORI2 '].mat'];
        if exist(cutFileName,'file')==0 % slices with single block
            cutFileName=[wholeSliceDir filesep casename '_' direction num2str(slInd) '[' ORI2 '].JPG'];
            cutMatFileName=[wholeSliceDir filesep 'cornersAndMasks' filesep casename '_' direction num2str(slInd) '[' ORI2 '].mat'];
        end
        
        disp('   Reading in volume of blockface photos and cut photo');
        BFS=myMRIread(blockFaceStackName);
        BF=uint8(squeeze(median(BFS.vol(:,:,max(1,SKIP):SKIP+2,:),3))); %uint8(squeeze(BFS.vol(:,:,200,:))); %
        BFMS=myMRIread(blockFaceStackMaskName);
        BFM=uint8(squeeze(median(BFMS.vol(:,:,max(1,SKIP):SKIP+2,:),3))); %uint8(squeeze(BFMS.vol(:,:,200,:))); %
        if i==length(d)  && structure_index<=2 % again, most posterior / lateral block
            BF=uint8(squeeze(median(BFS.vol(:,:,(end-SKIP-2):(end-2),:),3)));
            BFM=uint8(squeeze(median(BFMS.vol(:,:,(end-SKIP-2):(end-2),:),3)));
        end
        
        clear BFS BFMS
        BC=imread(cutFileName);
        load(cutMatFileName,'BW','Mframe','cornersIn','cornersOut','rescalingFactor');
        
        disp('   Resampling cut photo');
        if norm(cornersIn(:,1)-cornersIn(:,2)) > norm(cornersIn(:,1)-cornersIn(:,4))  % landscape
            SIZ=[90 120]/RES_C; % it's actually 120x90mm
        else  % portrait
            SIZ=[120 90]/RES_C;
        end
        cornersTarget=[1 1; 1 SIZ(2); SIZ(1) SIZ(2); SIZ(1) 1]';
        A=[cornersIn(2,:)' cornersIn(1,:)' zeros(4,2) ones(4,1) zeros(4,1);
            zeros(4,2) cornersIn(2,:)' cornersIn(1,:)'  zeros(4,1) ones(4,1)];
        b=[cornersTarget(2,:)'; cornersTarget(1,:)'];
        aux=A\b;
        tform=affine2d([aux(1) aux(3) 0; aux(2) aux(4) 0; aux(5) aux(6) 1]);
        BCw = imwarp(imresize(BC,1/rescalingFactor),tform,'OutputView',imref2d(SIZ));
        BWw = imwarp(BW,tform,'OutputView',imref2d(SIZ));
        BWw(BWw~=round(BWw))=0;
        BCwm=BCw; BCwm(repmat(BWw,[1 1 3])~=Bind)=0;

        
        % This is the scaling due to difference in resolution, plus tissue
        % shrinking due to embedding. We will registered to this image, and then
        % work out the transform that goes back to the original image
        scalingFactor = RES_C / RES_B * APPROX_SHRINK_FACTOR;
        BFs=imresize(BF,1/scalingFactor);
        BFMs=imresize(BFM,1/scalingFactor,'nearest');
        BFmaskedScaled=BFs;
        BFmaskedScaled(repmat(BFMs,[1 1 3])==0)=0;
        
        % Also, we crop the photograph of the blocks around the block of interest.
        % Once more, we need to be careful, to modify the final transform to
        % account for this
        margin=20;
        [BWwc,cropping]=cropLabelVol(BWw==Bind,margin); cropping(end)=3;
        BCwc=applyCropping(BCw,cropping);
        BCwc(repmat(BWwc,[1 1 3])==0)=0;
        
        disp('  Rigid registration');
        [optimizer,metric] = imregconfig('multimodal');
        optimizer.InitialRadius = 0.001;
        optimizer.Epsilon = 1.5e-4;
        optimizer.GrowthFactor = 1.01;
        optimizer.MaximumIterations = 300;
        
        moving=rgb2gray(BCwc); moving(moving==0)=nan;
        fixed=rgb2gray(BFmaskedScaled);
        
        close all
        figure
        subplot(2,3,1), imshow(BFmaskedScaled), title('FIXED');
        subplot(2,3,2), imshow(BCwc), title('MOVING (no rot)');
        subplot(2,3,3), imshow(imrotate(BCwc,90)), title('MOVING (rot 90)');
        subplot(2,3,5), imshow(imrotate(BCwc,-90)), title('MOVING (rot -90)');
        subplot(2,3,6), imshow(imrotate(BCwc,180)), title('MOVING (rot 180)');
        
        choice= menu('Choose an angle','0','90','-90','180');
        switch choice, case 1, theta=0; case 2, theta=pi/2; case 3, theta=-pi/2; case 4, theta=pi; end
        close all
        
        R=[cos(theta) sin(theta); -sin(theta) cos(theta)];
        t=-R*[size(moving,2); size(moving,1)]/2 + [size(fixed,2) ; size(fixed,1)]/2;
        tformIni=affine2d([R  t; 0 0 1]');
        
        % tformCS = imregtform(moving,fixed, 'similarity', optimizer, metric,'InitialTransformation',tformIni);
        % I now force it to be rigid. I'd rather leave the estimation of
        % the shrinking factor (beyond APPROX_SHRINK_FACTOR) to the linear
        % co-registratin of the blocks. This new approach is more robust
        tformCS = imregtform(moving,fixed, 'rigid', optimizer, metric,'InitialTransformation',tformIni);
        
        aux=[1 0 0; 0 1 0; -cropping(2)  -cropping(1) 1];
        aux=aux*tformCS.T*scalingFactor; aux(end)=1;
        tform=affine2d(aux);
        
        WARPED = imwarp(BCwm,tform,'OutputView',imref2d(size(BF)));
        shrinkingFactor=sqrt(det(tform.T)/(RES_C/RES_B)^2);
        
        close all
        for nview=1:10
            imshow(BF), pause(0.25);
            imshow(WARPED), pause(0.25);
        end
        
        happy = questdlg('Are you happy with the registration?',' ','Yes','No','Yes');
        while strcmp(happy,'No')
            uiwait(msgbox('Please select a minimum of 3 corresponding landmarks and close the window when ready','modal'));
            close all
            [movP,fixP] = cpselect(BCwc,BFs,'Wait',true);
            close all
            
            np=size(movP,1);
            if np<3
                uiwait(msgbox(['I said at least 3 points; you only provided ' num2str(np)],'modal'));
            else
                
                % tformIni = fitgeotrans(movP, fixP, 'similarity');
                % % Ensure it's similarity
                % aux=tformIni.T(1:2,1:2);
                % s=det(aux);
                % aux=aux/sqrt(s);
                % Y=(aux(2,1)-aux(1,2))/2;
                % X=(aux(1,1)+aux(2,2))/2;
                % theta=atan2(Y,X);
                % aux=sqrt(s)*[cos(theta) -sin(theta); sin(theta) cos(theta)];
                % tformIni.T(1:2,1:2)=aux;
                
                % Like I said earlier, I now force it to be rigid, and leave
                % the estimtion of the exact shrinking factor for later 
                aux=computeRigidTransformation( movP, fixP); % no RANSAC, all inliers
                tformIni=affine2d(aux');
             
                % refine registration
                % (and again, I switch to rigid)
                % tformCS = imregtform(moving,fixed, 'similarity', optimizer, metric,'InitialTransformation',tformIni);
                tformCS = imregtform(moving,fixed, 'rigid', optimizer, metric,'InitialTransformation',tformIni);
                
                aux=[1 0 0; 0 1 0; -cropping(2)  -cropping(1) 1];
                aux=aux*tformIni.T*scalingFactor; aux(end)=1;
                tformNoRef=affine2d(aux);
                
                aux=[1 0 0; 0 1 0; -cropping(2)  -cropping(1) 1];
                aux=aux*tformCS.T*scalingFactor; aux(end)=1;
                tform=affine2d(aux);
                
                WARPED = imwarp(BCwm,tform,'OutputView',imref2d(size(BF)));
                WARPEDnoref = imwarp(BCwm,tformNoRef,'OutputView',imref2d(size(BF)));
                shrinkingFactor=sqrt(det(tform.T)/(RES_C/RES_B)^2);
                shrinkingFactorNoref=sqrt(det(tformNoRef.T)/(RES_C/RES_B)^2);
                
                again='Yes';
                close all, figure(1)
                while strcmp(again,'Yes')
                    for nview=1:15
                        subplot(1,2,1), imshow(BF), title('Refined')
                        subplot(1,2,2), imshow(BF), title('Not refined');
                        pause(0.25);
                        subplot(1,2,1), imshow(WARPED), title('Refined')
                        subplot(1,2,2), imshow(WARPEDnoref), title('Not refined');
                        pause(0.25);
                    end
                    again = questdlg('Want to see registrations again?',' ','Yes','No','Yes');
                end
                close all
                
                happy = questdlg('Are you happy with the registration?',' ','With the refined one','With the *non*-refined one','No','With the refined one');
                if strcmp(happy,'With the *non*-refined one')
                    tform=tformNoRef;
                    WARPED=WARPEDnoref;
                end
            end
        end
        close all
        
        imwrite(BF,[outputDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.target.png']);
        imwrite(WARPED,[outputDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.warped.png']);
        % I could just not save this file, but I'll do it for now, for
        % compatibility with old code
        shrinkingFactor = APPROX_SHRINK_FACTOR;
        save([outputDir filesep casename '_' direction num2str(slice) '.' num2str(Bind) '.mat'],'tform','shrinkingFactor');
        
        disp('   Done! ');
        disp(' ');
    end
    disp('Structure done!');
    disp(' ');
    
end
disp('All done!!!!!');
