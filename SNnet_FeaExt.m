function [f BlkIdx] = SNnet_FeaExt(InImg,V,SNnet)
% =======INPUT=============
% InImg     Input images (cell)  
% V         given Semi-NMF filter banks (cell)
% SNnet    SNnet parameters (struct)
%       .SNnet.NumStages      
%           the number of stages in SNnet; e.g., 2  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., [5 3]
%           means patch size equalt to 5 and 3 in the first stage and second stage, respectively 
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize 
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio 
%           overlapped block region ratio; e.g., 0 means no overlapped 
%           between blocks, and 0.3 means 30% of blocksize is overlapped 
%       .Pyramid
%           spatial pyramid matching; e.g., [1 2 4], and [] if no Pyramid
%           is applied
% =======OUTPUT============
% f         SNnet features (each column corresponds to feature of each image)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

addpath('./Utils')

if length(SNnet.NumFilters)~= SNnet.NumStages;
    display('Length(SNnet.NumFilters)~=SNnet.NumStages')
    return
end

NumImg = length(InImg);

OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg;
for stage = 1:SNnet.NumStages
     [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
           SNnet.PatchSize(stage), SNnet.NumFilters(stage), V{stage});  
end

[f BlkIdx] = HashingHist(SNnet,ImgIdx,OutImg);
%





