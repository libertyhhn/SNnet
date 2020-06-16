% Title   : "Semi-NMF network for image classification",2019 Chinese Control Conference (CCC). 
% Authors : Huang, H., Yang, Z., Liang, N., & Li, Z
% Affl.   : Guangdong University of Technology, Guangzhou, China
% Email   : libertyhhn@foxmail.com
% 	
% Be noted that we modified from the Deep Semi-NMF implementation provide by the authors as follows:
% "PCANet: A simple deep learning baseline for image classification?"
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma,  
% IEEE Trans. Image Processing, vol. 24, no. 12, pp. 5017-5032, Dec. 2015. 
% PCANet code URL: http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
make; 

TrnSize = 100; 
ImgSize = 28; 
ImgFormat = 'gray'; %'color' or 'gray'

%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
 load('./MNISTdata/mnist_basic');
% load('./MNISTdata/MNIST_var'); 

% ===== Reshuffle the training data =====
% Randnidx = randperm(size(mnist_train,1)); 
% mnist_train = mnist_train(Randnidx,:); 
% =======================================

TrnData = mnist_train(1:TrnSize,1:end-1)';  % partition the data into training set and validation set
TrnLabels = mnist_train(1:TrnSize,end);
% ValData = mnist_train(TrnSize+1:end,1:end-1)';
% ValLabels = mnist_train(TrnSize+1:end,end);
clear mnist_train;
%TestData = mnist_test(100,1:end-1)';
%TestLabels = mnist_test(100,end);
TestData = mnist_test(1:5000,1:end-1)';
TestLabels = mnist_test(1:5000,end);
clear mnist_test;


% ==== Subsampling the Training and Testing sets ============
% (comment out the following four lines for a complete test) 
% TrnData = TrnData(:,1:4:end);  % sample around 2500 training samples
% TrnLabels = TrnLabels(1:4:end); % 
% TestData = TestData(:,1:50:end);  % sample around 1000 test samples  
% TestLabels = TestLabels(1:50:end); 
% ===========================================================

nTestImg = length(TestLabels);

%% SNnet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
SNnet.NumStages = 2;
SNnet.PatchSize = [7 7];
SNnet.NumFilters = [8 8];
SNnet.HistBlockSize = [7 7]; 
SNnet.BlkOverLapRatio = 0.5;
SNnet.Pyramid = [];

fprintf('\n ====== SNnet Parameters ======= \n')
SNnet

%% SNnet Training with 10000 samples

fprintf('\n ====== SNnet Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData; 
tic;
[ftrain V BlkIdx] = SNnet_train(TrnData_ImgCell,SNnet,1,TrnLabels); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 


%% SNnet Feature Extraction and Testing 

TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 

fprintf('\n ====== SNnet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
for idx = 1:1:nTestImg
    
    ftest = SNnet_FeaExt(TestData_ImgCell(idx),V,SNnet); % extract a test feature using trained PCANet model 

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm
   
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/100); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
    TestData_ImgCell{idx} = [];
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of SNnet, followed by a linear SVM classifier =====');
fprintf('\n     SNnet training time: %.2f secs.', SNnet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);




    