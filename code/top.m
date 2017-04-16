function top

% Get the list of categories
listing = dir('101_Objectcategories');
categories = extractfield(listing, 'name');
categories = categories(3:end);

% Read the image dataset
X = []; % All the images
y = [];   % Labeled images
for i = 1:length(categories)
    tmp = strcat('101_Objectcategories/', categories(i), '/*.jpg');
    imgsCurrDir = extractfield(dir(tmp{1}), 'name');
    for j= 1:length(imgsCurrDir)
        tmp = strcat('101_Objectcategories/', categories(i), '/', imgsCurrDir(j));
        X{end+1} = imread(tmp{1});
        y(end+1) = i;
    end
end

% Background subtraction using active contour


% Train the network - Forward & Backward pass


% Validate the network - Forward pass