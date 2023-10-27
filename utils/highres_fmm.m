function highres_fmm(block, subject, nslice, blockpath)

dirpath = [blockpath filesep 'MRI_aparc/'];
filename = [subject '_'  block  '_'  nslice];
%d = dir([dirpath filesep 'tmp' filesep subject '_'  block  '_'  nslice + '.mat']);

%for it_d=1:length(d)
%disp(['Processing section ' num2str(it_d) '/' num2str(length(d))]);

aparcFile = [dirpath filesep filename  '.nii.gz'];
load([dirpath filesep 'tmp' filesep filename  '.mat'], 'image', 'mask', 'labels', 'vox2ras');

mri = [];
mri.vox2ras0 = vox2ras;
mri.volres = sqrt(sum(mri.vox2ras0(1:3,1:3).^2));

[~, init_labels] = max(image,[],3);
W = 1 + 10000 * double(mask);

for n = 1 : length(labels)
    S = image(:,:,n) > 0.5; % región cortical and cortex, erosionada un poco
    [~,D] = imsegfmm(W, S, 0.0); % el threshold da igual, no se usa porque ignoras el primer argumento
    dist(:,:,n) = D;
end
dist(:, :, 1) = 10000000;
[~, label_fmm] = min(dist,[],3);
for n = 1 : length(labels)
    label_fmm(label_fmm == n) = labels(n);
end
final_labels = label_fmm.*mask;
final_labels = imrotate(final_labels, -90);
final_labels = fliplr(final_labels);
mri.vol = final_labels;
myMRIwrite(mri, aparcFile);

%clear dist
    
%end




