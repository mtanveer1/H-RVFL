% Please cite the following paper if you are using this code.
% Reference: Mushir Akhtar, Ritik Mishra, M. Sajid, A. Quadir, M. Tanveer, and Mohd. Arshad. "Advancing RVFL networks: Robust classification with the HawkEye loss function",
%            31st International Conference on Neural Information Processing (ICONIP), 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We have put a demo of the "H-RVFL" model with the "congressional_voting" dataset

clc;
clear;
warning off all;
format compact;

%% Data Preparation
addpath(genpath('C:\Users\mushi\OneDrive\Desktop\H-RVFL-GitHub-code'))
temp_data1=load('congressional_voting.mat');

temp_data=temp_data1.congressional_voting;
[d,e] = size(temp_data);
%define the class level +1 or -1
for i=1:d
    if temp_data(i,e)==0
        temp_data(i,e)=-1;
    end
end

X=temp_data(:,1:end-1); mean_X = mean(X,1); std_X = std(X);
X = bsxfun(@rdivide,X-repmat(mean_X,size(X,1),1),std_X);
All_Data=[X,temp_data(:,end)];

[samples,~]=size(All_Data);
split_ratio=0.8;
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
testX=testing_Data(:,1:end-1); testY=testing_Data(:,end);
trainX=training_Data(:,1:end-1); trainY=training_Data(:,end);

%% %% Hyperparameter range
% C=10.^[-6:2:6];  % Regularization parameter
% n= [3:20:203]    % numbner of hidded nodes
% a= 0.1:0.2:5;    % loss function parameters
% b= 0.1:0.2:5
% e= [0.001, 0.01, 0.1]
% We have tuned 6 Activation functions namely, sigmoid, sin, tribas, radbas, tansing, and relu.


option.C=1;
option.N= 63;
option.a= 0.9;
option.b= 1.9;
option.e= 0.01;
option.activation=1;

[Train_Accuracy, Test_Accuracy]   = H_RVFL_Function(trainX,trainY,testX,testY,option);

fprintf(1, 'Training Accuracy of H_RVFL model is: %f\n', Train_Accuracy);
fprintf(1, 'Testing Accuracy of H_RVFL model is: %f\n', Test_Accuracy);
