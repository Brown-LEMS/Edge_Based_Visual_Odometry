% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/edges_on_imgs.m
% Load undistorted images saved by the pipeline (edges are detected on undistorted images)
outputs_path = "../outputs_euroc/";

img_0_left  = imread(outputs_path + "left_undistorted_frame_0.png");
img_0_right = imread(outputs_path + "right_undistorted_frame_0.png");

data_path = outputs_path + "finalized_stereo_edge_pairs_frame_0.txt";

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

% Select 100 random correspondences
num_edges = size(left_x, 1);
num_to_draw = min(100, num_edges);
random_indices = randperm(num_edges, num_to_draw);

% Generate a distinct color per correspondence
colors = hsv(num_to_draw);

% Setup the figure
figure('Name', 'Stereo Edge Matches', 'Position', [100, 100, 1200, 500]);
imshow(img_combined);
hold on;
title('Stereo Edge Pairs: 100 Random Correspondences');

for i = 1:num_to_draw
    idx = random_indices(i);
    c = colors(i, :);
    lx = left_x(idx);
    ly = left_y(idx);
    rx = right_x_shifted(idx);
    ry = right_y(idx);
    plot(lx, ly, '.', 'Color', c, 'MarkerSize', 10);
    plot(rx, ry, '.', 'Color', c, 'MarkerSize', 10);
    line([lx, rx], [ly, ry], 'Color', [c, 0.5], 'LineWidth', 0.8);
end

hold off;