function [ent] = entropy_f(TrainData,n_channels,n_trials)
%ENTROPY_F Summary of this function goes here
%   Detailed explanation goes here
ent = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        ent(channel, i) = entropy(X(i, :));
    end
end
ent = normalize(ent);
end
