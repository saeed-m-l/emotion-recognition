function [ps] = psr(TrainData,n_channels,n_trials,fr1,fr2)
%PSR Summary of this function goes here
%   Detailed explanation goes here
fs = 256;
ps = zeros(n_channels,n_trials);
for channel = 1: 1: n_channels   
    X = squeeze(TrainData(channel, :, :)).';
    for i = 1: 1: n_trials
        x=(X(i, :));
        frequenciesOfInterest = [fr1, fr2];  % Frequency band between 10 Hz and 20 Hz
        [pxx, f] = pwelch(x, [], [], [], fs);
        freqIndices = f >= frequenciesOfInterest(1) & f <= frequenciesOfInterest(2);
        s_w = pxx(freqIndices, :);
        ps(channel, i) = sum(abs(s_w))/sum(abs(pxx));
    end
end
ps = normalize(ps);
end
