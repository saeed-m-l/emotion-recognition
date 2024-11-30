function [ratio] = ratio_d(TrainData,n_channels,n_trials)
%RATIO_D Summary of this function goes here
%   Detailed explanation goes here
fs = 256;
ratio = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        x=(X(i, :));
        df_dt = diff(x) ./ (1/fs);
        ratio_dy_dt = mean(abs(df_dt)) / mean(abs(x));
        ratio(channel, i) = ratio_dy_dt;
    end
end
ratio = normalize(ratio);
end
