function [variance] = variance_f(TrainData,n_channels,n_trials)
%VARIANCE_F Summary of this function goes here
%   Detailed explanation goes here
variance = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        variance(channel, i) = var(X(i, :));
    end
end
variance = normalize(variance);
end

