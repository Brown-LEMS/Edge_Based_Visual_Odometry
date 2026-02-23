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

% Load images
kf_img = imread(img_path_0_left);
cf_img = imread(img_path_1_left);

% KF Edge data
kf_edge_idx = 36817;
kf_location = [592.009, 275.146];  % KF edge location on keyframe image

% GT data on CF
gt_location_cf = [608.463, 270.621];
gt_orientation_cf_rad = 2.96273;  % radians
gt_orientation_cf_deg = rad2deg(gt_orientation_cf_rad);  % convert to degrees

% Veridical CF Edges data
cf_edges_locations = [
    607.501, 270.561;
    608.001, 270.52;
    608.499, 270.474;
    608.995, 270.426
];

cf_edges_orientations_deg = [
    175.868;
    174.839;
    174.328;
    174.257
];

% Create figure with subplots
figure('Position', [100, 100, 1200, 600]);

% Subplot 1: Keyframe image with KF edge
subplot(1, 2, 1);
imshow(kf_img);
title('Keyframe Image (KF Edge)');
hold on;

% Plot KF edge location
plot(kf_location(1), kf_location(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

text(50, 50, sprintf('KF Edge Index: %d\nLocation: (%.1f, %.1f)', kf_edge_idx, kf_location(1), kf_location(2)), 'Color', 'white', 'FontSize', 12, 'BackgroundColor', 'black');

% Subplot 2: Current frame image with GT location and veridical edges
subplot(1, 2, 2);
imshow(cf_img);
title('Current Frame Image (CF Edges)');
hold on;

% Plot GT location on CF
plot(gt_location_cf(1), gt_location_cf(2), 'go', 'MarkerSize', 12, 'LineWidth', 3);
% Plot GT orientation arrow
arrow_length = 30;
quiver(gt_location_cf(1), gt_location_cf(2), ...
       arrow_length * cosd(gt_orientation_cf_deg), ...
       arrow_length * sind(gt_orientation_cf_deg), ...
       'g', 'LineWidth', 3, 'MaxHeadSize', 0.8);

% Plot veridical CF edges
colors = ['r', 'b', 'm', 'c'];  % Different colors for each edge
for i = 1:size(cf_edges_locations, 1)
    % Plot edge location
    plot(cf_edges_locations(i, 1), cf_edges_locations(i, 2), ...
         [colors(i) 'o'], 'MarkerSize', 8, 'LineWidth', 2);
    
    % Plot edge orientation arrow
    quiver(cf_edges_locations(i, 1), cf_edges_locations(i, 2), ...
           arrow_length * cosd(cf_edges_orientations_deg(i)), ...
           arrow_length * sind(cf_edges_orientations_deg(i)), ...
           colors(i), 'LineWidth', 2, 'MaxHeadSize', 0.5);
end

% Add legend
legend_entries = {'GT Location', 'GT Orientation'};
for i = 1:size(cf_edges_locations, 1)
    legend_entries = [legend_entries, sprintf('CF Edge %d', i), sprintf('CF Orientation %d', i)];
end
legend(legend_entries, 'Location', 'northeastoutside');

% Add text information
text_info = sprintf(['KF Edge: %d (%.1f, %.1f)\nGT Location: (%.1f, %.1f)\nGT Orientation: %.1f°\n%d Veridical CF Edges'], ...
                   kf_edge_idx, kf_location(1), kf_location(2), gt_location_cf(1), gt_location_cf(2), gt_orientation_cf_deg, size(cf_edges_locations, 1));
text(50, 50, text_info, 'Color', 'white', 'FontSize', 10, 'BackgroundColor', 'black');

hold off;