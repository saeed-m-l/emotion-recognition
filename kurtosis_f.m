function [kur] = kurtosis_f(TrainData,n_channels,n_trials)
%KURTOSIS Summary of this function goes here
%   Detailed explanation goes here
kur = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        kur(channel, i) = var(X(i, :));
    end
end
kur = normalize(kur);
end
