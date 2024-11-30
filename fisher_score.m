function [fisher] = fisher_score(feature,n_channels,TrainLabel)
%FISHER_SCORE Summary of this function goes here
%   Detailed explanation goes here
fisher = zeros(n_channels,1);
for channel = 1: 1: n_channels   
    x1 = find(TrainLabel == 1);
    x2 = find(TrainLabel == -1);
    mu0 = sum(feature(channel, :)) / length(feature);
    mu1 = sum(feature(channel, x1)) / length(x1);
    mu2 = sum(feature(channel, x2)) / length(x2);
    var1 = var(feature(channel, x1));
    var2 = var(feature(channel, x2));
    fisher(channel) = ((mu0 - mu1)^2 + (mu0 - mu2)^2) / (var1 + var2); 
end
end

