%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading the Training Data %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[x,y, yNT] = importBankData('bank-full.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading the Test Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
[xTest,yTest, yNTTest] = importBankData('bank.csv');

%% normalize X
xNorm = mapminmax(x);
xTestNorm = mapminmax(xTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Logistic Regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Train %%%%%
xLR = [xNorm; ones(1,length(xNorm))];

%% Set the intital theta vector to all zeros
theta = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

%% Initial value of Hypothesis
h = logsig(theta*xLR);

m = length(y);
alpha = 0.1;

[finalThetaLR, J] = trainLR(xLR, y, theta, alpha, 0.05, 10000);

%%%%% Test %%%%%
xLRTest = [xTestNorm; ones(1,length(xTestNorm))];
resultLR = hardlim(logsig(finalThetaLR*xLRTest) - 0.5);
hold on
figure(1);
confusionMatrixLR = plotconfusion(yTest, resultLR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Train %%%%%
net = newff(xNorm, yNT, 10, {'tansig' 'logsig'}, 'traingd', 'learngd','mse', {}, {}, '');
net.trainParam.lr= 0.3; 
net.trainParam.goal = 0.05;
net.trainParam.show = 100; 
net.trainParam.epochs = 10000; 
net = train(net, xNorm, yNT);

%%%%% Test %%%%%
resultNN = hardlim(net(xTestNorm) - 0.5);
figure(2);
confusionMatrixNN = plotconfusion(yNTTest, resultNN);
