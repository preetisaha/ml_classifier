function [ finalTheta, J ] = trainLR(X, y, initialTheta, alpha, threshold, epochs  )
theta = initialTheta;
h = logsig(theta*X);
m = length(y);
cost = -sum(y.*log(h)+(1-y).*(log(1-h)))/m;
costs = [];
costs = [costs cost];
count = 1;

while (cost > threshold && count < epochs )
    theta = theta -(alpha/m)*(h-y)*X';
    h = logsig(theta*X);
    cost = -sum(y.*log(h)+(1-y).*(log(1-h)))/m;
    costs = [costs cost];
    count = count + 1;
end

finalTheta = theta;
J = costs;
end