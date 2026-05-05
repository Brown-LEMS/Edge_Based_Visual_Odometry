% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/edges_on_imgs.m
% Setup paths to dataset
source_dataset_folder = "/oscar/data/bkimia/Datasets/EuRoC/";
dataset_sequence_path = "MH_01_easy/mav0/";

% For frame 0 
stereo_pair_0 = "cam0/data/1403636579763555584.png";
img_path_0_left = source_dataset_folder + dataset_sequence_path + stereo_pair_0;
img_0_left = imread(img_path_0_left);
stereo_pair_1 = "cam1/data/1403636579763555584.png";
img_path_0_right = source_dataset_folder + dataset_sequence_path + stereo_pair_1;
img_0_right = imread(img_path_0_right);

data_path = '../outputs_euroc/finalized_stereo_edge_pairs_frame_1.txt';

% Parse the text file
% readmatrix automatically handles the delimiter and we skip the first header line.
edge_data = readmatrix(data_path, 'NumHeaderLines', 1);

% Extract coordinates based on the data format provided
% Col 1: left_x, Col 2: left_y
left_x = edge_data(:, 1);
left_y = edge_data(:, 2);

% Col 4: right_x, Col 5: right_y
right_x = edge_data(:, 4);
right_y = edge_data(:, 5);

% To draw lines connecting pairs on both images, we combine them side-by-side (montage)
img_combined = [img_0_left, img_0_right];
img_width = size(img_0_left, 2);

% Shift the right image's X-coordinates by the width of the left image
right_x_shifted = right_x + img_width;

% Setup the figure
figure('Name', 'Stereo Edge Matches', 'Position', [100, 100, 1200, 500]);
imshow(img_combined);
hold on;
title('Stereo Edge Pairs: All Edges, 100 Random Connections');

% Plot ALL left and right edges on their respective halves
plot(left_x, left_y, 'r.', 'MarkerSize', 10); % Left edges in Red
plot(right_x_shifted, right_y, 'g.', 'MarkerSize', 10); % Right edges in Green

% --- Select 100 random lines to draw ---
num_edges = size(left_x, 1);
num_lines_to_draw = min(100, num_edges); % Safeguard if there are fewer than 100 edges
random_indices = randperm(num_edges, num_lines_to_draw);

% Filter coordinates for the lines using the random indices
X_lines_subset = [left_x(random_indices)'; right_x_shifted(random_indices)'];
Y_lines_subset = [left_y(random_indices)'; right_y(random_indices)'];

% Draw the random subset of connecting lines
line(X_lines_subset, Y_lines_subset, 'Color', [1 1 0 0.5], 'LineWidth', 0.8); % Yellow lines

hold off;