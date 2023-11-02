% This function computes Gaussian scale space features of an image, up to a 
% maximum order, and at a given vector of scales. Works with grayscale & RGB
%  F = computeGaussianScaleSpaceFeatures(IMAGE,max_order,vector_of_scales)
function F=computeGaussianScaleSpaceFeatures(Ir,featMaxDerivOrder,featScales)

% count number of features (to allocate feature matrix)
count=0;
for order=0:featMaxDerivOrder, for ox=0:order, for oy=0:order, if ox+oy==order, count=count+1; end; end; end; end
nfeats=size(Ir,3)*count*length(featScales);

% in case...
Ir=double(Ir);

% compute features
F=zeros([size(Ir,1) size(Ir,2) nfeats]);

indF=0;
for s=1:length(featScales)
    if featScales(s)==0
        IrB=double(Ir);
    else
        IrB=imfilter(double(Ir),fspecial('gaussian',1+2*ceil(3*featScales(s))*[1 1],featScales(s)));
    end
    
    for order=0:featMaxDerivOrder
        for ox=0:order
            for oy=0:order
                if ox+oy==order
                    for c=1:size(Ir,3) % r,g,b channels
                        indF=indF+1;
                        feat=IrB(:,:,c);
                        for t=1:ox
                            feat=imfilter(feat,[-1 0 1]);
                        end
                        for t=1:oy
                            feat=imfilter(feat,[-1 0 1]');
                        end
                        F(:,:,indF)=feat;
                    end
                end
            end
        end
    end
end
