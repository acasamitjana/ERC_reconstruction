% This function downsamples an MRI struct in memory by a given factor
function mri2 = downsampleMRI(mri,factors,mode)

if nargin<3
    mode='linear';
end

if length(factors)==1
    factors=repmat(factors,[1 3]);
end

if strcmp(mode,'nearest')  % smooth unless we're doing labels
    S=mri.vol;
else
    S=GaussFilt3d(mri.vol,0.4*factors);
end

[II,JJ,KK]=ndgrid(.5+factors(1)/2:factors(1):size(S,1),.5+factors(2)/2:factors(2):size(S,2),.5+factors(3)/2:factors(3):size(S,3));
I=interpn(S,II,JJ,KK,mode);

mri2=[];
mri2.vol=I;
mri2.volres=mri.volres.*[factors(2) factors(1) factors(3)];
mri2.vox2ras0=mri.vox2ras0;
mri2.vox2ras0(1:3,1:3)=mri.vox2ras0(1:3,1:3).*repmat([factors(2) factors(1) factors(3)],[3 1]);
mri2.vox2ras0(1:3,4)=mri2.vox2ras0(1:3,4)+mri.vox2ras0(1:3,1:3)*([factors(2); factors(1); factors(3)]/2-0.5);


