function [reshapedMatrix] = featureCreator(concatenatedMatrix,binaryMatrix,n_selection)
%FEATURECREATOR Summary of this function goes here
%   Detailed explanation goes here
s = size(concatenatedMatrix);
%reshapedMatrix = zeros(s(1),s(2),n_selection);
reshapedMatrix = zeros(s(1),n_selection);
for x=1:1:s(1)
    tmp=1;
    for y=1:1:s(2)
        %tmp=1;
        for o=1:1:s(3)
            if binaryMatrix(y,o) == 1
                reshapedMatrix(x,tmp) = concatenatedMatrix(x,y,o);
                tmp = tmp+1;
            end
        end
    end
end
%reshapedMatrix = reshape(reshapedMatrix,s(1),s(2)*n_selection);
%reshapedMatrix = reshape(reshapedMatrix,n_selection);
end

