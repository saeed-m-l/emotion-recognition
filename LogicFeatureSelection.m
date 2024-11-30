function [binaryMatrix] = LogicFeatureSelection(fisherMatrix,numTopAttributes)
%LOGICFEATURESELECTION Summary of this function goes here
%   Detailed explanation goes here
%{
binaryMatrix = zeros(size(fisherMatrix));

for channel = 1:size(fisherMatrix, 1)
    [~, sortedIndices] = sort(fisherMatrix(channel, :), 'descend');
    binaryMatrix(channel, sortedIndices(1:numTopAttributes)) = 1;
end
%}
[~, indices] = sort(fisherMatrix(:), 'descend');
top5Indices = indices(1:numTopAttributes);

binaryMatrix = zeros(size(fisherMatrix));
binaryMatrix(top5Indices) = 1;
end

