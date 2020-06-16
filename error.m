T = sum(sum((Rx - V*V'*Rx).^2))
A = V*V'*Rx

% clear; clc;
% load ('');
% % train_path='..\Data\TrainingSet\';
% phi=zeros(64*64,20); 
% for i=1:20 path=strcat(train_path,num2str(i),'.bmp'); 
%     Image=imread(path); 
%     Image=imresize(Image,[64,64]); 
%     phi(:,i)=double(reshape(Image,1,[])'); 
% end; %mean 
% mean_phi=mean(phi,2); 
% mean_face=reshape(mean_phi,64,64); 
% Image_mean=mat2gray(mean_face); 
% imwrite(Image_mean,'meanface.bmp','bmp'); 
% %demean 
% for i=1:19 
%     X(:,i)=phi(:,i)-mean_phi; 
% end
% Lx=X'*X; 
% tic;
% [eigenvector,eigenvalue]=eigs(Lx,19); 
% toc; 
% %normalization 
% for i=1:19 
%     %K-L±ä»» 
%     UL(:,i)=X*eigenvector(:,i)/sqrt(eigenvalue(i,i)); 
% end
% %display Eigenface 
% for i=1:19 
%     Eigenface=reshape(UL(:,i),[64,64]); 
%     figure(i); 
%     imshow(mat2gray(Eigenface)); 
% end




