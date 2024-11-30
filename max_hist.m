function [max_h] = max_hist(TrainData,n_channels,n_trials)
%MAX_HIST Summary of this function goes here
%   Detailed explanation goes here
max_h = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        x=(X(i, :));
        numBins = 20;
        histValues = hist(x, numBins);
        [maxValue, max_index] = max(histValues);
        max_h(channel, i) = max_index;
    end
end
max_h = normalize(max_h);
end
