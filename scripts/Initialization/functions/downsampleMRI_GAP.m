% This function downsamples an MRI struct in memory by a given factor
function mri2 = downsampleMRI_GAP(mri, inPlaneFactors)

inPlaneSiz = round([size(mri.vol,1) size(mri.vol,2)] ./ inPlaneFactors);
inPlaneFactors=[size(mri.vol,1) size(mri.vol,2)]./inPlaneSiz; % account for rounding errors
outPlaneSize = size(mri.vol,3);
mri2 = [];
mri2.vol = zeros([inPlaneSiz outPlaneSize]);
for j=1:size(mri.vol,3)
    mri2.vol(:,:,j)=imresize(mri.vol(:,:,j),inPlaneSiz);
end

mri2.volres=mri.volres.*[inPlaneFactors 1];
mri2.vox2ras0=mri.vox2ras0;
mri2.vox2ras0(1:3,1)=mri2.vox2ras0(1:3,1)*inPlaneFactors(1);
mri2.vox2ras0(1:3,2)=mri2.vox2ras0(1:3,2)*inPlaneFactors(2);
aux=(inPlaneFactors-1)./(2*inPlaneFactors);
mri2.vox2ras0(1:3,4)=mri2.vox2ras0(1:3,4) - mri2.vox2ras0(1:3,1:3) * [aux'; 0];

