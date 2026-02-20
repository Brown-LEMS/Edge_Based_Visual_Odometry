% Visualization of KF and CF edges
% KF Edge Index: 7860 on Frame 0 Left at (328.959, 197.173)
% KF Edge Index: 7860 on Frame 0 Right at (321.477, 197.173)
% CF Edges: 7829, 7830, 7831, 7867, 7868

% Setup paths to dataset
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";

% For frame 0 (keyframe)
stereo_pair_0 = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
img_path_0_left = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im0.png";
img_0_left = imread(img_path_0_left);
img_path_0_right = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im1.png";
img_0_right = imread(img_path_0_right);

% For frame 1 (current frame)
stereo_pair_1 = "images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png";
img_path_1_right = source_dataset_folder + dataset_sequence_path + stereo_pair_1 + "/im1.png";
img_1_right = imread(img_path_1_right);

% KF edge data
kf_edge_loc_left = [328.95962637, 197.17279013];
kf_edge_loc_right = [321.477, 197.173];
kf_edge_idx = 7860;

% CF edge data (Idx, X, Y, Orientation in degrees)
cf_edges = [
    7829, 335.396, 191.554, -170.531;
    7830, 335.824, 191.645, -170.531;
    7831, 336.286, 191.730, -171.356;
    7867, 336.622, 191.803, -172.178;
    7868, 336.986, 191.855, -172.178;
];

% GT projected location and orientation (in radians)
gt_location = [336.220, 191.709];
gt_orientation = -2.93114;

% Create figure with 1x3 subplots
figure('Position', [100, 100, 2000, 600]);

% Left subplot: Frame 0 Left with KF edge
subplot(1, 3, 1);
imshow(img_0_left);
hold on;
title(sprintf('Frame 0 (Keyframe) - Left Image\nEdge Index: %d', kf_edge_idx), 'FontSize', 14, 'FontWeight', 'bold');

% Draw KF edge as a line with orientation
edge_length = 40;
kf_ori_rad = atan2(sin(gt_orientation), cos(gt_orientation));
dx = edge_length/2 * cos(kf_ori_rad);
dy = edge_length/2 * sin(kf_ori_rad);

% Draw edge line
line([kf_edge_loc_left(1)-dx, kf_edge_loc_left(1)+dx], [kf_edge_loc_left(2)-dy, kf_edge_loc_left(2)+dy], ...
     'Color', 'red', 'LineWidth', 4);

% Draw point at center
plot(kf_edge_loc_left(1), kf_edge_loc_left(2), 'ro', 'MarkerSize', 15, 'LineWidth', 3);

% Add arrow to show orientation
arrow_len = edge_length/3;
arrow_dx = arrow_len * cos(kf_ori_rad);
arrow_dy = arrow_len * sin(kf_ori_rad);
quiver(kf_edge_loc_left(1), kf_edge_loc_left(2), arrow_dx, arrow_dy, 0, ...
       'Color', 'red', 'LineWidth', 2.5, 'MaxHeadSize', 1.5);

% Add text annotation
text(kf_edge_loc_left(1)+35, kf_edge_loc_left(2)-20, sprintf('KF Edge #%d\n(%.2f, %.2f)', kf_edge_idx, kf_edge_loc_left(1), kf_edge_loc_left(2)), ...
     'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

text(20, 30, sprintf('Frame 0 Left Image'), 'Color', 'white', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Middle subplot: Frame 0 Right with KF edge
subplot(1, 3, 2);
imshow(img_0_right);
hold on;
title(sprintf('Frame 0 (Keyframe) - Right Image\nEdge Index: %d', kf_edge_idx), 'FontSize', 14, 'FontWeight', 'bold');

% Draw edge line
line([kf_edge_loc_right(1)-dx, kf_edge_loc_right(1)+dx], [kf_edge_loc_right(2)-dy, kf_edge_loc_right(2)+dy], ...
     'Color', 'red', 'LineWidth', 4);

% Draw point at center
plot(kf_edge_loc_right(1), kf_edge_loc_right(2), 'ro', 'MarkerSize', 15, 'LineWidth', 3);

% Add arrow to show orientation
quiver(kf_edge_loc_right(1), kf_edge_loc_right(2), arrow_dx, arrow_dy, 0, ...
       'Color', 'red', 'LineWidth', 2.5, 'MaxHeadSize', 1.5);

% Add text annotation
text(kf_edge_loc_right(1)+35, kf_edge_loc_right(2)-20, sprintf('KF Edge #%d\n(%.2f, %.2f)', kf_edge_idx, kf_edge_loc_right(1), kf_edge_loc_right(2)), ...
     'Color', 'red', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Add location text at corner
text(20, 30, sprintf('Frame 0 Right Image'), 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Right subplot: Frame 1 Right with CF edges and GT location
subplot(1, 3, 3);
imshow(img_1_right);
hold on;
title('Frame 1 (Current) - Right Image', 'FontSize', 14, 'FontWeight', 'bold');

% Plot GT projected location as a line
gt_ori_rad = gt_orientation;  % Already in radians
gt_dx = edge_length/2 * cos(gt_ori_rad);
gt_dy = edge_length/2 * sin(gt_ori_rad);

% Draw GT projected edge line (dashed)
line([gt_location(1)-gt_dx, gt_location(1)+gt_dx], [gt_location(2)-gt_dy, gt_location(2)+gt_dy], ...
     'Color', 'blue', 'LineWidth', 3, 'LineStyle', '--');

% Draw point at GT location
plot(gt_location(1), gt_location(2), 'b^', 'MarkerSize', 20, 'LineWidth', 2.5);

% Add arrow for GT orientation
arrow_len = edge_length/3;
arrow_dx = arrow_len * cos(gt_ori_rad);
arrow_dy = arrow_len * sin(gt_ori_rad);
quiver(gt_location(1), gt_location(2), arrow_dx, arrow_dy, 0, ...
       'Color', 'blue', 'LineWidth', 2, 'MaxHeadSize', 1.2, 'LineStyle', '--');

text(gt_location(1)+35, gt_location(2)+15, sprintf('GT Projected\n(%.3f, %.3f)\nOri: %.3f', gt_location(1), gt_location(2), gt_orientation), ...
     'Color', 'blue', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Plot CF edges as lines (Using 5 distinct colors for 5 edges)
colors = {'g', 'm', 'c', 'y', [1 0.5 0]}; % Green, Magenta, Cyan, Yellow, Orange
for i = 1:size(cf_edges, 1)
    idx = cf_edges(i, 1);
    x = cf_edges(i, 2);
    y = cf_edges(i, 3);
    ori_deg = cf_edges(i, 4);
    ori_rad = deg2rad(ori_deg);  % Convert degrees to radians
    
    % Calculate line endpoints
    dx = edge_length/2 * cos(ori_rad);
    dy = edge_length/2 * sin(ori_rad);
    
    % Draw edge line
    line([x-dx, x+dx], [y-dy, y+dy], 'Color', colors{i}, 'LineWidth', 4);
    
    % Draw point at center
    plot(x, y, 'o', 'Color', colors{i}, 'MarkerSize', 14, 'LineWidth', 2.5);
    
    % Add arrow to show orientation
    arrow_len = edge_length/3;
    arrow_dx = arrow_len * cos(ori_rad);
    arrow_dy = arrow_len * sin(ori_rad);
    quiver(x, y, arrow_dx, arrow_dy, 0, 'Color', colors{i}, 'LineWidth', 2.5, 'MaxHeadSize', 1.3);
    
    % Distance to GT
    dist_to_gt = sqrt((x - gt_location(1))^2 + (y - gt_location(2))^2);
    
    % Add text annotation
    text(x+35, y-20-i*35, sprintf('CF Edge #%d\n(%.2f, %.2f)\nOri: %.1f° | Dist: %.2f px', idx, x, y, ori_deg, dist_to_gt), ...
         'Color', colors{i}, 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);
end

% Add legend/info
text(20, 30, sprintf('Frame 1 Right Image - CF Candidates'), 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
text(20, 60, sprintf('Blue(dash): GT | Grn: #7829 | Mag: #7830 | Cya: #7831 | Yel: #7867 | Org: #7868'), 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

hold off;

% Overall title
sgtitle(sprintf('Edge-Based Visual Odometry: KF Edge #%d Temporal Projection\nKF Right Location: (%.3f, %.3f) | GT CF Location: (%.3f, %.3f)', ...
                kf_edge_idx, kf_edge_loc_right(1), kf_edge_loc_right(2), gt_location(1), gt_location(2)), ...
        'FontSize', 16, 'FontWeight', 'bold');

% Customize figure
set(gcf, 'Color', 'w');

% Save figure
saveas(gcf, 'kf_cf_edge_visualization.png');
fprintf('\n=== Visualization Summary ===\n');
fprintf('KF Edge #%d:\n', kf_edge_idx);
fprintf('  Left Image Location:  (%.3f, %.3f)\n', kf_edge_loc_left(1), kf_edge_loc_left(2));
fprintf('  Right Image Location: (%.3f, %.3f)\n', kf_edge_loc_right(1), kf_edge_loc_right(2));
fprintf('\nGT Projected on CF:\n');
fprintf('  Location: (%.3f, %.3f)\n', gt_location(1), gt_location(2));
fprintf('  Orientation: %.5f\n', gt_orientation);
fprintf('\nCF Veridical Edges Found:\n');
for i = 1:size(cf_edges, 1)
    idx = cf_edges(i, 1);
    x = cf_edges(i, 2);
    y = cf_edges(i, 3);
    ori = cf_edges(i, 4);
    dist_to_gt = sqrt((x - gt_location(1))^2 + (y - gt_location(2))^2);
    fprintf('  Edge #%d: (%.3f, %.3f), Ori: %.1f°, Dist to GT: %.3f px\n', idx, x, y, ori, dist_to_gt);
end
fprintf('\nFigure saved as: kf_cf_edge_visualization.png\n');
fprintf('===============================\n\n');