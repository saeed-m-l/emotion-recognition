function [hurst] = Hurst(TrainData,n_channels,n_trials)
%HURST Summary of this function goes here
%   Detailed explanation goes here
hurst = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        data = X(i, :);
        n = length(data);
        R = zeros(1, n);
    
        for ii = 1:n
            cumsumData = cumsum(data - mean(data));
            rangeData = max(cumsumData(1:ii)) - min(cumsumData(1:ii));
            stdDevData = std(data(1:ii));
            R(i) = rangeData / stdDevData;
        end
        logN = log(1:n);
        logR = log(R);
        fitCoefficients = polyfit(logN, logR, 1);
        hurstExponent = fitCoefficients(1);
        hurst(channel, i) = hurstExponent;
     end
end
hurst = normalize(hurst);
end
