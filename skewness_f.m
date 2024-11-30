function [sk] = skewness_f(TrainData,n_channels,n_trials)
%SKEWNESS_F Summary of this function goes here
%   Detailed explanation goes here
sk = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        x=(X(i, :));
        sk(channel, i) = skewness(x);
    end
end
sk = normalize(sk);
end
