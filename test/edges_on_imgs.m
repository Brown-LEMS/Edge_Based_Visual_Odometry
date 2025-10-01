% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/edges_on_imgs.m
% Setup paths to dataset
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";

% For frame 0 (previous frame)
stereo_pair_0 = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
img_path_0_left = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im0.png";
img_0_left = imread(img_path_0_left);
img_path_0_right = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im1.png";
img_0_right = imread(img_path_0_right);

% For frame 1 (current frame)
stereo_pair_1 = "images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png";
img_path_1_left = source_dataset_folder + dataset_sequence_path + stereo_pair_1 + "/im0.png";
img_1_left = imread(img_path_1_left);
img_path_1_right = source_dataset_folder + dataset_sequence_path + stereo_pair_1 + "/im1.png";
img_1_right = imread(img_path_1_right);

% Extract edge data for a specific previous edge (e.g., edge index 8)
extract_edge_data('/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/outputs/ncc_scores_gt_debug_frame_1.csv', 8);

% Load the extracted data
prev = importdata("prev.txt");
toed_left = importdata("ncc.txt");
gt = importdata("gt.txt");

% Load third-order edges
third_order_edges_0 = importdata("/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/third_order_edges_frame_0.txt");
third_order_edges_1 = importdata("/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/third_order_edges_frame_1.txt");

% Calculate direction vectors for visualization
% Previous edge direction
prev_cos = cos(prev(3));
prev_sin = sin(prev(3));
prev_vec = [prev_cos, prev_sin];

% Candidate edges directions
dirs_cos = cos(toed_left(:,3)); 
dirs_sin = sin(toed_left(:,3));
dir_vecs = [dirs_cos, dirs_sin];

% Ground truth edge direction
gt_cos = cos(gt(3));
gt_sin = sin(gt(3));
gt_vec = [gt_cos, gt_sin];

% Third-order edges directions (frame 0)
third_order_dirs_cos_0 = cos(third_order_edges_0(:,3));
third_order_dirs_sin_0 = sin(third_order_edges_0(:,3));
third_order_dir_vecs_0 = [third_order_dirs_cos_0, third_order_dirs_sin_0];

% Third-order edges directions (frame 1)
third_order_dirs_cos_1 = cos(third_order_edges_1(:,3));
third_order_dirs_sin_1 = sin(third_order_edges_1(:,3));
third_order_dir_vecs_1 = [third_order_dirs_cos_1, third_order_dirs_sin_1];

% Create figure with 2 subplots side by side
figure('Position', [100, 100, 1800, 800]);

% Left subplot: Previous frame with the edge and third-order edges
subplot(1, 2, 1);
imshow(img_0_left); hold on;
title('Previous Frame (Frame 0)', 'FontSize', 14);

% Plot all third-order edges from frame 0
plot(third_order_edges_0(:,1), third_order_edges_0(:,2), 'y.', 'MarkerSize', 6);
quiver(third_order_edges_0(:,1), third_order_edges_0(:,2), third_order_dir_vecs_0(:,1), third_order_dir_vecs_0(:,2), 0, 'y-', 'LineWidth', 0.8);

% Plot the specific edge we're tracking
plot(prev(1), prev(2), 'g*', 'MarkerSize', 10);
quiver(prev(1), prev(2), prev_vec(1), prev_vec(2), 0, 'g-', 'LineWidth', 2);
text(prev(1)+10, prev(2), ['Edge ' num2str(8)], 'Color', 'green', 'FontSize', 12);

% Add text showing number of edges
text(20, 20, sprintf('Third-Order Edges (Frame 0): %d', size(third_order_edges_0, 1)), 'Color', 'yellow', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);

hold off;

% Right subplot: Current frame with GT, candidates, and third-order edges
subplot(1, 2, 2);
imshow(img_1_left); hold on;
title('Current Frame (Frame 1)', 'FontSize', 14);

% Plot third-order edges from frame 1
plot(third_order_edges_1(:,1), third_order_edges_1(:,2), 'y.', 'MarkerSize', 6);
quiver(third_order_edges_1(:,1), third_order_edges_1(:,2), third_order_dir_vecs_1(:,1), third_order_dir_vecs_1(:,2), 0, 'y-', 'LineWidth', 0.8);

% Plot all candidates with their directions
plot(toed_left(:,1), toed_left(:,2), 'c.', 'MarkerSize', 8); 
quiver(toed_left(:,1), toed_left(:,2), dir_vecs(:,1), dir_vecs(:,2), 0, 'c-', 'LineWidth', 1.2);

% Plot the ground truth edge
plot(gt(1), gt(2), 'r*', 'MarkerSize', 10);
quiver(gt(1), gt(2), gt_vec(1), gt_vec(2), 0, 'r-', 'LineWidth', 2);
text(gt(1)+10, gt(2), 'GT', 'Color', 'red', 'FontSize', 12);

% Add legend for right subplot
legend('Third-Order Edges', '', 'Candidates', '', 'Ground Truth', '', 'Location', 'southwest');

% Add text showing number of edges
text(20, 20, sprintf('Third-Order Edges (Frame 1): %d', size(third_order_edges_1, 1)), 'Color', 'yellow', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);
text(20, 40, sprintf('Candidates: %d', size(toed_left, 1)), 'Color', 'cyan', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);

% Customize figure appearance
set(gcf, 'Color', 'w');
sgtitle(['Visualization for Edge Index ' num2str(8) ' with Third-Order Edges'], 'FontSize', 16);

% Save the figure
saveas(gcf, ['both_frames_edge_' num2str(8) '_visualization.png']);
fprintf('Both frames visualization saved as both_frames_edge_%d_visualization.png\n', 8);

% Create additional figure showing only third-order edges side by side
figure('Position', [100, 100, 1800, 800]);

% Left subplot: Previous frame third-order edges only
subplot(1, 2, 1);
imshow(img_0_left); hold on;
title('Frame 0 - Third-Order Edges Only', 'FontSize', 14);
plot(third_order_edges_0(:,1), third_order_edges_0(:,2), 'm.', 'MarkerSize', 6);
quiver(third_order_edges_0(:,1), third_order_edges_0(:,2), third_order_dir_vecs_0(:,1), third_order_dir_vecs_0(:,2), 0, 'm-', 'LineWidth', 0.8);
text(20, 20, sprintf('Third-Order Edges (Frame 0): %d', size(third_order_edges_0, 1)), 'Color', 'magenta', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);

% Right subplot: Current frame third-order edges only
subplot(1, 2, 2);
imshow(img_1_left); hold on;
title('Frame 1 - Third-Order Edges Only', 'FontSize', 14);
plot(third_order_edges_1(:,1), third_order_edges_1(:,2), 'm.', 'MarkerSize', 6);
quiver(third_order_edges_1(:,1), third_order_edges_1(:,2), third_order_dir_vecs_1(:,1), third_order_dir_vecs_1(:,2), 0, 'm-', 'LineWidth', 0.8);
text(20, 20, sprintf('Third-Order Edges (Frame 1): %d', size(third_order_edges_1, 1)), 'Color', 'magenta', 'FontSize', 12, 'BackgroundColor', [0 0 0 0.5]);

% Customize figure appearance
set(gcf, 'Color', 'w');
sgtitle('Third-Order Edges Comparison Between Frames', 'FontSize', 16);

% Save the figure
saveas(gcf, 'third_order_edges_comparison.png');
fprintf('Third-order edges comparison saved as third_order_edges_comparison.png\n');