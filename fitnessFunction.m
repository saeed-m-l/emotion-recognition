function [fit] = fitnessFunction(data,indices,labels)
%FITNESSFUNCTION Summary of this function goes here
%   Detailed explanation goes here
penalty = unique(indices);
penalty = length(penalty)/20;
% disp(indices)
X = data(:,unique(indices));
x1 = labels == 1;
x2 = labels == -1;
X1 = X(x1,:);
X2 = X(x2,:);
S1 = sum(sum((X1-mean(X1,2)).^2));
S2 = sum(sum((X2-mean(X2,2)).^2));
fit = S1+S2-penalty;
end

