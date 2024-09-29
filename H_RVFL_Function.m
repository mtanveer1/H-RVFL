function [Train_Accuracy, Test_Accuracy]  = H_RVFL_Function(trainX,trainY,testX,testY,option)

%%%%%%%%%%%%%%%%% Training Starts %%%%%%%%%%%%%%%%%
N = option.N;
C = option.C;
s = 1;
activation = option.activation;


m=2^5;          % mini batch size
a=option.a;     % a, b, and e are loss parameter
b=option.b;
e=option.e;
%%%%%%%%%%%%%%%%%
alltrain=[trainX,trainY];
l=size(alltrain,1);
% Set the random seed for reproducibility
seed = 0;
rng(seed);

% Get the number of rows in the matrix
numRows = size(alltrain, 1);

% Generate a random permutation of row indices
permIndices = randperm(numRows);

% Interchange rows of the matrix based on the permutation
randomizedMatrix = alltrain(permIndices, :);

rand_data=randomizedMatrix(1:m,:);

trainXrand=rand_data(:,1:end-1);
trainYrand=rand_data(:,end);

[Nsample,Nfea] = size(trainXrand);



W = (rand(Nfea,N)*2*s-1);
bias = s*rand(1,N);
X1 = trainXrand*W+repmat(bias,Nsample,1);

if activation == 1
    X1 = sigmoid(X1);
elseif activation == 2
    X1 = sin(X1);
elseif activation == 3
    X1 = tribas(X1);
elseif activation == 4
    X1 = radbas(X1);
elseif activation == 5
    X1 = tansig(X1);
elseif activation == 6
    X1 = relu(X1);
end


X = [trainXrand,X1]; %Direct Link
X = [X,ones(Nsample,1)];%bias in the output layer

%%%%%%%%%%%%%%%%%%%%%%%%%

max_iter = 500;  % maximum iteration number
tol = 10^-6;
eta0=0.01; %initial learning rate
beta = 0.01*ones(size(X,2),1); %initialize model parameter
v=0.01*ones(size(X,2),1); % initialize velocity
k= 0.1;  % learning decay rate factor
r= 0.6;  % momentum parameter
betaPrevious = inf;

for t=1:max_iter


    u=X*beta-trainYrand; %Xi matrix with respect to all samples
    temp3=zeros(size(X,2),1);

    for i=1:m
        if u(i)< -e %here e is epsilon
            temp2= b *a^2 *X(1,:)' * exp( a * ( u(i,:) + e) ) * ( u(i,:) + e);
        elseif u(i) >= -e && u(i) <= e
            temp2= zeros(size(X,2),1);
            % temp2= zeros(size(X(1,:)));
        elseif u(i) > -e
            temp2= b * a^2 *X(1,:)' *exp( -a * ( u(i,:) - e ) ) * ( u(i,:) - e);
        end

        temp3=temp3+temp2;
    end

    temp4=C*temp3;
    gradient=beta+temp4;


    beta=beta+r*v;
    v=r*v-eta0*gradient;
    beta=beta+v;
    eta0=eta0*exp(-k*t);




    if norm(beta-betaPrevious)<tol
        fprintf('Converged at iteration %d\n', t);
        break
    else
        betaPrevious=beta;
    end


end



Predict_Y_train = sign(X*beta); %output of H-RVFL

Train_Accuracy = Evaluate(trainYrand,Predict_Y_train);




%%%%%%%%%%%%%%%%%%%% Testing Starts %%%%%%%%%%%%%%%%%%%%%



Nsample = size(testX,1);


X1 = testX*W+repmat(bias,Nsample,1);

if activation == 1
    X1 = sigmoid(X1);
elseif activation == 2
    X1 = sin(X1);
elseif activation == 3
    X1 = tribas(X1);
elseif activation == 4
    X1 = radbas(X1);
elseif activation == 5
    X1 = tansig(X1);
elseif activation == 6
    X1 = relu(X1);
end


X = [testX,X1];

X=[X,ones(Nsample,1)];

rawScore = X*beta;
f=sign(rawScore);


Test_Accuracy = Evaluate(testY,f);
end

