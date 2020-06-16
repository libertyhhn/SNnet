  function [V,Rx] = GsemiNMF_FilterBank(InImg, PatchSize, NumFilters) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.��NMF��ά��
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% =========================

addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

% NMF parameters



err_rat=0.025;
eps=1e-9;

maxiter=200;
r=NumFilters;
%% Learning PCA filters (V)
NumChls = size(InImg{1},3);%��ɫͨ���ĸ���
%Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2); % sum covariance
Rx = [];
%ÿ��ͼ���зֿ����?
for i = RandIdx %1:ImgZ
    im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]);% collect all the patches of the ith image in a matrix, and perform patch mean removal
    B = im(:);
    Rx = [Rx,B];
    %Rx = Rx + im*im'; % sum of all the input images' covariance matrix
end
[a,b]=size(im);
 [m, N]=size(Rx); % V contains your data in its column vectors
 W0=rand(m,r); % randomly initialize basis
 W0=W0*diag(1./sum(W0));
 H0=rand(r,N);
 W=W0;
 W1=zeros(a,b,r);
 W2=zeros(b,a,r);
 W3=zeros(a,r);
 H=H0;
 F_obj(1,1)=sum(sum((Rx-W*H).*(Rx-W*H)))/sum(sum(Rx.*Rx)); %ֵΪ0-1֮��ŷ�����?���Ϊ��ӦԪ����ˣ�sumΪÿ��Ԫ�����?
 X = Rx;
 %for iter=1:maxiter
%     
%     H=H.*(W'*Rx+eps)./(W'*W*H+eps);
%     W=W.*(Rx*H'+eps)./(W*H*H'+eps);
%     W=W*diag(1./sum(W,1));
%W = X * pinv(H);%更新Z矩阵  
  
%A = W' * X;%求Z的转置乘以X  
%Ap = (abs(A)+A)./2;  
%An = (abs(A)-A)./2;  
  
%B = W' * W;%Z^T*Z  
%Bp = (abs(B)+B)./2;  
%Bn = (abs(B)-B)./2;  
  
%H = H .* sqrt((Ap + Bn * H) ./ max(An + Bp * H, eps));%更新H矩阵总公�?
% load FERET_crop_database;
load PIE_pose27;
fa_label = gnd;
W = ssnmf(Rx,fa_label,r);
%     F_obj(1,iter+1)=sum(sum((Rx-W*H).*(Rx-W*H)))/sum(sum(Rx.*Rx));
%     if F_obj(iter+1)<err_rat
%         break;
 %     end
 %end
W1 = reshape(W,a,b,r);
W2 = permute(W1,[2 1 3]);
W3 = reshape(sum(W2),a,r);
V = W3;
% Rx = Rx/(NumRSamples*size(im,2));
% [E D] = eig(Rx); %��������ֵ������������eig,RxΪЭ�������?
% [~, ind] = sort(diag(D),'descend');
% V = E(:,ind(1:NumFilters));  % principal eigenvectors 



 



