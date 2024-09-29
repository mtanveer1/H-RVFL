# H-RVFL: Advancing RVFL networks: Robust classification with the HawkEye loss function

Please cite the following paper if you are using this code. 

Reference: Mushir Akhtar, Ritik Mishra, M. Sajid, A. Quadir, M. Tanveer, and Mohd. Arshad. "Advancing RVFL networks: Robust classification with the HawkEye loss function", 31st International Conference on Neural Information Processing (ICONIP), 2024.

This paper incorporates the HawkEye loss (H-loss) function into the RVFL framework. The H-loss function features nice mathematical properties, including smoothness and boundedness, while simultaneously incorporating an insensitive zone. Each characteristic brings its own advantages: 1) Boundedness limits the impact of extreme errors, enhancing robustness against outliers; 2) Smoothness facilitates the use of gradient based optimization algorithms, ensuring stable and efficient convergence; and 3) The insensitive zone mitigates the effect of minor discrepancies and noise. Notably, this work addresses a significant gap, as no bounded loss function has been incorporated into RVFL to date.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

We have put a demo of the “H-RVFL” model with the “congressional_voting” dataset.


Description of files:

Main: This is the main file to run selected models on datasets. In the path variable specify the path to the folder containing the codes and datasets on which you wish to run the algorithm.
HE_RVFL_Function: The main file calls this file for the training and testing process. This contain the function for HE-RVFL model solved using NAG algorithm.
Evaluate.m: This function computes the accuracy.


%% %% Hyperparameter range
% C=10.^[-6:2:6];  % Regularization parameter
% n= [3:20:203]    % number of hidded nodes
% a= 0.1:0.2:5;    % loss function parameters
% b= 0.1:0.2:5
% e= [0.001, 0.01, 0.1]
% We have tuned 6 Activation functions namely, sigmoid, sin, tribas, radbas, tansing, and relu.


The codes are not optimized for efficiency. The codes have been cleaned for better readability and documented and are not exactly the same as used in our paper. For the detailed experimental setup, please follow the paper. We have re-run and checked the codes only in a few datasets, so if you find any bugs/issues, please write to Mushir Akhtar (phd2101241004@iiti.ac.in).
