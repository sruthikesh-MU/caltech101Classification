%function top

clear all; close all;
% Get the list of categories
listing = dir('101_ObjectCategories/');
categories = extractfield(listing, 'name');
categories = categories(3:end);

% Read the image dataset
XCell = []; % All the images
y = [];   % Labeled images
for i = 1:length(categories)
    tmp = strcat('101_ObjectCategories/', categories(i), '/*.jpg');
    imgsCurrDir = extractfield(dir(tmp{1}), 'name');
    for j= 1:length(imgsCurrDir)
        tmp = strcat('101_ObjectCategories/', categories(i), '/', imgsCurrDir(j));
        img = imread(tmp{1});
        if ndims(img)==2
            img = cat(3, img, img, img);
        end
        XCell{end+1} = imresize(img, [96, 96]);
        y(end+1) = i;
    end
end

X = uint8(zeros([size(XCell{1}), length(XCell)]));
for i = 1:length(XCell)
    X(:, :, :, i) = XCell{i};
end

% imshow(X(1,:,:,:))

% Image pre-processing
% Background subtraction using active contour

% Active contour has to be on gray scale image
mask = uint8(zeros(size(rgb2gray(X{600}))));
mask(25:end-25,25:end-25) = 1;
bw = activecontour(rgb2gray(X{600}), mask, 100);

% Split data
trainingNumFiles = length(XCell);
rng(1) % For reproducibility
%[trainDigitData, testDigitData] = splitEachLabel(X, 0.7,'randomize');
trainRatio = 0.8;
testRatio = 1-trainRatio;
valRatio = 1-(trainRatio+ testRatio);
[trainInd, valInd, testInd] = dividerand(trainingNumFiles, trainRatio, valRatio, testRatio);

trainData = X(:, :, :, trainInd(:));
testData = X(:, :, :, testInd(:));
valData = [];

% NN 
% Train the network - Forward & Backward pass
layers = [imageInputLayer([96 96 3]);
          convolution2dLayer(5,20, 'stride', 2);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,100);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(101);
          softmaxLayer();
          classificationLayer()];   % Could be regressionLayer also

trainLabels = uint8(y(trainInd(:)));

options = trainingOptions('sgdm','MaxEpochs', 30, 'InitialLearnRate',0.0001);   
net = trainNetwork(trainData, categorical(trainLabels), layers, options);

% Validate the network - Forward pass

% SVM Based classification