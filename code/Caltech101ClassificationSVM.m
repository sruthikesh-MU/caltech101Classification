function Caltech101ClassificationSVM

rootFolder = fullfile('101_ObjectCategories');

% Get the list of categories
listing = dir('101_ObjectCategories/');
categories = extractfield(listing, 'name');
categories = categories(3:end);

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);

layers = [imageInputLayer([96 96 3]);
          convolution2dLayer(5,20, 'stride', 2);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,100);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(102);
          softmaxLayer();
          classificationLayer()];   % Could be regressionLayer also

 imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
 
 [trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
 options = trainingOptions('sgdm','MaxEpochs', 30, 'InitialLearnRate',0.0001);   
 net = trainNetwork(imds, layers, options);
 
YTest = classify(net, testSet);
TTest = testSet.Labels;

accuracy = sum(YTest == TTest)/numel(TTest)

end
 
function Iout = readAndPreprocessImage(filename)

    I = imread(filename);

    % Some images may be grayscale. Replicate the image 3 times to
    % create an RGB image.
    if ismatrix(I)
        I = cat(3,I,I,I);
    end

    % Resize the image as required for the CNN.
    Iout = imresize(I, [96 96]);

    % Note that the aspect ratio is not preserved. In Caltech 101, the
    % object of interest is centered in the image and occupies a
    % majority of the image scene. Therefore, preserving the aspect
    % ratio is not critical. However, for other data sets, it may prove
    % beneficial to preserve the aspect ratio of the original image
    % when resizing.
end

