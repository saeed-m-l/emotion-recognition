%%%% SAEED MANSUR LAKURAJ %%%% 99102304 %%%%
%% Load Data
close all;clear;clc;
Fs = 256;
data = load('Project_data.mat');
%%
trainData = data.TrainData;
trainLabel = data.TrainLabels;
testData = data.TestData;
n_channels = 59;
n_trials_train = 550;
n_trials_test = 159;
%% Feature Extraction ( Part 1 for selection) , I will select 5 attributes from here
% variance
var_f = variance_f(trainData, n_channels, n_trials_train);
var_fish = fisher_score(var_f, n_channels, trainLabel);

% max_h
maxH = max_hist(trainData, n_channels, n_trials_train);
maxH_fish = fisher_score(maxH, n_channels, trainLabel);
% kurt
kurt = kurtosis_f(trainData, n_channels, n_trials_train);
kurt_fish = fisher_score(kurt, n_channels, trainLabel);

% skeweness
sk = skewness_f(trainData, n_channels, n_trials_train);
sk_fish = fisher_score(sk, n_channels, trainLabel);

% ratio_dy_dt
ratio = ratio_d(trainData, n_channels, n_trials_train);
ratio_fish = fisher_score(ratio, n_channels, trainLabel);

% average & median & max
avg = zeros(n_channels, n_trials_train);
mx = zeros(n_channels, n_trials_train);
med = zeros(n_channels, n_trials_train);
for channel = 1: 1: n_channels
    X = squeeze(trainData(channel, :, :)).';
    for i = 1: 1: n_trials_train
        avg(channel, i) = mean(X(i, :));
        med(channel, i) = median(X(i, :));
        mx(channel, i) = max(X(i, :));
    end
end
avg = normalize(avg);
med = normalize(med);
mx = normalize(mx);
avg_fish = fisher_score(avg, n_channels, trainLabel);
med_fish = fisher_score(med, n_channels, trainLabel);
mx_fish = fisher_score(mx, n_channels, trainLabel);
% entropy
ent = skewness_f(trainData, n_channels, n_trials_train);
ent_fish = fisher_score(ent, n_channels, trainLabel);

% fish matrix for part 1
matrix_fish1 = [var_fish, maxH_fish,kurt_fish, sk_fish, ratio_fish, avg_fish, med_fish, mx_fish, ent_fish];

% Create logic matrix

logic_feature1 = LogicFeatureSelection(matrix_fish1, 200);

feature3_1 = cat(3,var_f.',maxH.',kurt.', sk.', ratio.',avg.',med.',mx.',ent.');
featureM1_train = featureCreator(feature3_1, logic_feature1,20);
%% Feature Extraction ( Part 2 for selection) , I will select 7 attributes from here 
% Some easy in calculation ferequency attributes
avg_f = zeros(n_channels, n_trials_train);
med_f = zeros(n_channels, n_trials_train);
bandP = zeros(n_channels, n_trials_train);
band99 = zeros(n_channels, n_trials_train);
for channel = 1: 1: n_channels
    X = squeeze(trainData(channel, :, :)).';
    for i = 1: 1: n_trials_train
        avg_f(channel, i) = meanfreq(X(i, :));
        med_f(channel, i) = medfreq(X(i, :));
        bandP(channel, i) = bandpower(X(i, :));
        band99(channel, i) = obw(X(i, :));
    end
end
avg_f = normalize(avg_f);
med_f = normalize(med_f);
bandP = normalize(bandP);
band99 = normalize(band99);                         
avgF_fish = fisher_score(avg_f, n_channels, trainLabel);
medF_fish = fisher_score(med_f, n_channels, trainLabel);
bandP_fish = fisher_score(bandP, n_channels, trainLabel);
band99_fish = fisher_score(band99, n_channels, trainLabel);

% Power spectral density for different bands ( 7 bands)
psr1 = psr(trainData, n_channels, n_trials_train,0.1,3);
psr1_fish = fisher_score(psr1, n_channels, trainLabel);

psr2 = psr(trainData, n_channels, n_trials_train,4,7);
psr2_fish = fisher_score(psr2, n_channels, trainLabel);

psr3 = psr(trainData, n_channels, n_trials_train,8,12);
psr3_fish = fisher_score(psr3, n_channels, trainLabel);

psr4 = psr(trainData, n_channels, n_trials_train,12,15);
psr4_fish = fisher_score(psr4, n_channels, trainLabel);

psr5 = psr(trainData, n_channels, n_trials_train,16,20);
psr5_fish = fisher_score(psr5, n_channels, trainLabel);

psr6 = psr(trainData, n_channels, n_trials_train,21,30);
psr6_fish = fisher_score(psr1, n_channels, trainLabel);

psr7 = psr(trainData, n_channels, n_trials_train,30,100);
psr7_fish = fisher_score(psr7, n_channels, trainLabel);


% fish matrix for part 2

matrix_fish2 = [avgF_fish,medF_fish,bandP_fish, band99_fish, psr1_fish, psr2_fish,psr3_fish, psr4_fish, psr5_fish, psr6_fish, psr7_fish];



% Create logic matrix

logic_feature2 = LogicFeatureSelection(matrix_fish2, 400);

feature3 = cat(3,avg_f.',med_f.',bandP.', band99.', psr1.', psr2.',psr3.', psr4.', psr5.', psr6.', psr7.');
featureM2_train = featureCreator(feature3, logic_feature2,40);
save('feautureM2_train.mat','featureM2_train')
%% Create X_test
% variance
var_f = variance_f(testData, n_channels, n_trials_test);
% max_h
maxH = max_hist(testData, n_channels, n_trials_test);
% kurt
kurt = kurtosis_f(testData, n_channels, n_trials_test);

% skeweness
sk = skewness_f(testData, n_channels, n_trials_test);

% ratio_dy_dt
ratio = ratio_d(testData, n_channels, n_trials_test);

% average & median & max
avg = zeros(n_channels, n_trials_test);
mx = zeros(n_channels, n_trials_test);
med = zeros(n_channels, n_trials_test);
for channel = 1: 1: n_channels
    X = squeeze(testData(channel, :, :)).';
    for i = 1: 1: n_trials_test
        avg(channel, i) = mean(X(i, :));
        med(channel, i) = median(X(i, :));
        mx(channel, i) = max(X(i, :));
    end
end
avg = normalize(avg);
med = normalize(med);
mx  = normalize( mx);
% entropy
ent = skewness_f(testData, n_channels, n_trials_test);

% Create logic matrix

feature1_test = cat(3,var_f.',maxH.',kurt.', sk.', ratio.',avg.',med.',mx.',ent.');

featureM1_test = featureCreator(feature1_test, logic_feature1,200);

% Some easy in calculation ferequency attributes
avg_f = zeros(n_channels, n_trials_test);
med_f = zeros(n_channels, n_trials_test);
bandP = zeros(n_channels, n_trials_test);
band99 = zeros(n_channels, n_trials_test);
for channel = 1: 1: n_channels
    X = squeeze(testData(channel, :, :)).';
    for i = 1: 1: n_trials_test
        avg_f(channel, i) = meanfreq(X(i, :));
        med_f(channel, i) = medfreq(X(i, :));
        bandP(channel, i) = bandpower(X(i, :));
        band99(channel, i) = obw(X(i, :)); 
    end
end
avg_f = normalize(avg_f);
med_f = normalize(med_f);
bandP = normalize(bandP);
band99 = normalize(band99);
% Power spectral density for different bands ( 7 bands)

psr1 = psr(testData, n_channels, n_trials_test,0.1,3);
psr2 = psr(testData, n_channels, n_trials_test,4,7);
psr3 = psr(testData, n_channels, n_trials_test,8,12);
psr4 = psr(testData, n_channels, n_trials_test,12,15);
psr5 = psr(testData, n_channels, n_trials_test,16,20);
psr6 = psr(testData, n_channels, n_trials_test,21,30);
psr7 = psr(testData, n_channels, n_trials_test,30,100);

% Create logic matrix

feature2_test = cat(3,avg_f.',med_f.',bandP.', band99.', psr1.', psr2.',psr3.', psr4.', psr5.', psr6.', psr7.');
featureM2_test = featureCreator(feature2_test, logic_feature2,400);
%% Train and Test X
%X_train = cat(2,featureM1_train,featureM2_train);
%save('X_train.mat','X_train')
X_test = cat(2,featureM1_test, featureM2_test);
save('X_test.mat','X_test')

%% MLP phase 1
% I DID this part in python
%% RBF phase 1
% I did this part in python
%% %%%%%%%%%%%%% PHASE 2 %%%%%%%%%%%%% %%
% GENETIC AlGorithm
% At first choose first 1000 features using fisher score
%%%%%%%%%%%%%%% WARNING %%%%%%%%%%%%%%%
% while running this i change attributes for group one from 20 to 200 and
% group 2 from 60 to 400
X_train_gen = cat(2,featureM1_train,featureM2_train);
save('X_train_gen.mat','X_train_gen')
%%

X_test_gen = cat(2,featureM1_test, featureM2_test);
save('X_test_gen.mat','X_test_gen')
%%
% I did next in python

%% ANN Part
% I Did this in python