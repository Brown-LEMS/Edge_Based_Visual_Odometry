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
% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/edges_on_imgs.m
% Setup paths to dataset
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";

% Specific edge coordinates to visualize
edge_left_f0 = [655.87, 438.46];   % Edge in left image frame 0
edge_right_f0 = [627.438, 443.823]; % Edge in right image frame 0

% Frame 1 projected edge (from your EBVO system)
edge_right_f1_projected = [641.939, 438.414]; % Projected edge in right image frame 1

% Frame 1 left image edges (candidate matches)
edges_frame1_left = [
    659.59 437.24; 658.55 437.65; 658.99 437.48; 659.43 437.31; 657.07 438.22;
    657.52 438.05; 657.96 437.89; 655.56 438.71; 656.02 438.58; 656.48 438.43;
    656.93 438.28; 654.04 439.13; 654.50 439.01; 654.97 438.89; 655.43 438.76;
    652.07 439.68; 652.52 439.55; 652.98 439.43; 653.44 439.30; 651.00 439.99;
    651.45 439.86; 638.05 443.65; 638.51 443.53; 638.97 443.41; 636.06 444.17;
    636.52 444.06; 636.98 443.95; 637.43 443.83; 633.54 444.67; 634.02 444.59;
    634.50 444.50; 634.97 444.41; 635.43 444.32; 631.03 445.14; 631.51 445.04;
    631.99 444.95; 632.47 444.86; 632.95 444.77; 630.45 445.30; 649.03 440.64;
    649.50 440.48; 649.95 440.33; 647.53 441.18; 648.00 441.03; 648.47 440.86;
    646.01 441.68; 646.50 441.54; 646.99 441.38; 644.01 442.16; 644.50 442.06;
    645.00 441.95; 645.50 441.83; 641.55 442.70; 642.02 442.59; 642.50 442.48;
    642.99 442.38; 643.48 442.28; 650.54 440.13; 657.33 450.02; 657.11 450.47;
    656.95 451.02; 656.93 451.53; 657.00 452.00; 657.07 452.47; 657.12 452.96;
    657.12 453.47; 657.07 453.99; 656.98 454.50; 656.88 454.98; 656.75 455.57;
    656.78 455.44; 656.61 456.04; 656.46 456.48; 656.14 457.10; 656.30 456.88;
    655.92 457.43; 655.09 458.11; 655.41 457.90; 654.08 458.64; 654.48 458.46;
    654.84 458.29; 652.57 459.15; 653.01 459.02; 653.45 458.89; 653.87 458.75;
    651.05 459.59; 651.48 459.46; 651.92 459.34; 650.41 459.85
];

% Create figure with 2x2 subplots for complete stereo view
figure('Position', [50, 50, 2400, 1200]);

% Top left: Frame 0 Left image with specific edge
subplot(2, 2, 1);
imshow(img_0_left); hold on;
title('Frame 0 - Left Image (Keyframe)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot the specific edge in left image frame 0
plot(edge_left_f0(1), edge_left_f0(2), 'ro', 'MarkerSize', 15, 'LineWidth', 4);
plot(edge_left_f0(1), edge_left_f0(2), 'r*', 'MarkerSize', 20, 'LineWidth', 3);

% Add crosshair
line([edge_left_f0(1)-15, edge_left_f0(1)+15], [edge_left_f0(2), edge_left_f0(2)], 'Color', 'red', 'LineWidth', 3);
line([edge_left_f0(1), edge_left_f0(1)], [edge_left_f0(2)-15, edge_left_f0(2)+15], 'Color', 'red', 'LineWidth', 3);

% Add text annotation
text(edge_left_f0(1)+20, edge_left_f0(2)-20, sprintf('KF Edge\n(%.2f, %.2f)', edge_left_f0(1), edge_left_f0(2)), ...
     'Color', 'red', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

text(20, 20, 'Keyframe Left', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Top right: Frame 0 Right image with specific edge
subplot(2, 2, 2);
imshow(img_0_right); hold on;
title('Frame 0 - Right Image (Keyframe)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot the specific edge in right image
plot(edge_right_f0(1), edge_right_f0(2), 'bo', 'MarkerSize', 15, 'LineWidth', 4);
plot(edge_right_f0(1), edge_right_f0(2), 'b*', 'MarkerSize', 20, 'LineWidth', 3);

% Add crosshair
line([edge_right_f0(1)-15, edge_right_f0(1)+15], [edge_right_f0(2), edge_right_f0(2)], 'Color', 'blue', 'LineWidth', 3);
line([edge_right_f0(1), edge_right_f0(1)], [edge_right_f0(2)-15, edge_right_f0(2)+15], 'Color', 'blue', 'LineWidth', 3);

% Add text annotation
text(edge_right_f0(1)+20, edge_right_f0(2)-20, sprintf('KF Stereo\n(%.3f, %.3f)', edge_right_f0(1), edge_right_f0(2)), ...
     'Color', 'blue', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Calculate and display disparity
disparity_f0 = edge_left_f0(1) - edge_right_f0(1);
text(20, 20, sprintf('F0 Disparity: %.3f px', disparity_f0), ...
     'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

text(20, 45, 'Keyframe Right', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Bottom left: Frame 1 Left image with multiple edges
subplot(2, 2, 3);
imshow(img_1_left); hold on;
title('Frame 1 - Left Image (Current)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot all frame 1 edges
plot(edges_frame1_left(:,1), edges_frame1_left(:,2), 'go', 'MarkerSize', 6, 'LineWidth', 2);
plot(edges_frame1_left(:,1), edges_frame1_left(:,2), 'g+', 'MarkerSize', 8, 'LineWidth', 2);

% Highlight some specific edges
highlighted_indices = [1, 10, 20, 30, 40, 50];
colors = {'r', 'm', 'c', 'y', 'k', 'w'};

for i = 1:length(highlighted_indices)
    idx = highlighted_indices(i);
    if idx <= size(edges_frame1_left, 1)
        plot(edges_frame1_left(idx,1), edges_frame1_left(idx,2), 'o', 'Color', colors{i}, 'MarkerSize', 12, 'LineWidth', 3);
        text(edges_frame1_left(idx,1)+8, edges_frame1_left(idx,2)-8, sprintf('%d', idx), ...
             'Color', colors{i}, 'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
    end
end

% Show Frame 0 edge location for reference
plot(edge_left_f0(1), edge_left_f0(2), 'rx', 'MarkerSize', 15, 'LineWidth', 4);
text(edge_left_f0(1)+10, edge_left_f0(2)+15, 'F0 Ref', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.8]);

text(20, 20, sprintf('CF Candidates: %d', size(edges_frame1_left, 1)), ...
     'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

text(20, 45, 'Current Frame Left', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Bottom right: Frame 1 Right image with projected edge
subplot(2, 2, 4);
imshow(img_1_right); hold on;
title('Frame 1 - Right Image (Current)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot the projected edge in Frame 1 Right image
plot(edge_right_f1_projected(1), edge_right_f1_projected(2), 'mo', 'MarkerSize', 15, 'LineWidth', 4);
plot(edge_right_f1_projected(1), edge_right_f1_projected(2), 'm*', 'MarkerSize', 20, 'LineWidth', 3);

% Add crosshair for projected edge
line([edge_right_f1_projected(1)-15, edge_right_f1_projected(1)+15], [edge_right_f1_projected(2), edge_right_f1_projected(2)], 'Color', 'magenta', 'LineWidth', 3);
line([edge_right_f1_projected(1), edge_right_f1_projected(1)], [edge_right_f1_projected(2)-15, edge_right_f1_projected(2)+15], 'Color', 'magenta', 'LineWidth', 3);

% Add text annotation for projected edge
text(edge_right_f1_projected(1)+20, edge_right_f1_projected(2)-20, sprintf('Projected\n(%.3f, %.3f)', edge_right_f1_projected(1), edge_right_f1_projected(2)), ...
     'Color', 'magenta', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Show Frame 0 right edge location for reference
plot(edge_right_f0(1), edge_right_f0(2), 'bx', 'MarkerSize', 15, 'LineWidth', 3);
text(edge_right_f0(1)+10, edge_right_f0(2)+15, 'F0 Ref', 'Color', 'blue', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.8]);

% Calculate motion and disparity for Frame 1
motion_x = edge_right_f1_projected(1) - edge_right_f0(1);
motion_y = edge_right_f1_projected(2) - edge_right_f0(2);
text(20, 20, sprintf('Motion: (%.1f, %.1f)', motion_x, motion_y), ...
     'Color', 'cyan', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

% Estimate disparity in Frame 1 (if we had a corresponding left edge)
if ~isempty(edges_frame1_left)
    estimated_disparity_f1 = edges_frame1_left(1,1) - edge_right_f1_projected(1);
    text(20, 45, sprintf('Est. F1 Disp: %.1f px', estimated_disparity_f1), ...
         'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
end

text(20, 70, 'Current Frame Right', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
hold off;

% Overall figure customization
set(gcf, 'Color', 'w');
sgtitle(sprintf('Edge-Based Visual Odometry: Frame 0 → Frame 1\nF0 Disparity: %.3f px | Projected: (%.3f, %.3f)', ...
                disparity_f0, edge_right_f1_projected(1), edge_right_f1_projected(2)), ...
        'FontSize', 16, 'FontWeight', 'bold');

% Save the comprehensive figure
saveas(gcf, 'ebvo_edge_tracking_analysis.png');
fprintf('EBVO edge tracking analysis saved as ebvo_edge_tracking_analysis.png\n');

% Create a detailed comparison view around the projected edge
figure('Position', [100, 100, 2000, 1000]);

% Define zoom window around the projected edge
zoom_size = 80;
center_x = edge_right_f1_projected(1);
center_y = edge_right_f1_projected(2);

% Calculate crop ranges
x_range = max(1, center_x-zoom_size):min(size(img_1_right, 2), center_x+zoom_size);
y_range = max(1, center_y-zoom_size):min(size(img_1_right, 1), center_y+zoom_size);

% Subplot 1: Frame 0 Left (zoomed around original edge)
subplot(2, 2, 1);
f0_left_x_range = max(1, edge_left_f0(1)-zoom_size):min(size(img_0_left, 2), edge_left_f0(1)+zoom_size);
f0_left_y_range = max(1, edge_left_f0(2)-zoom_size):min(size(img_0_left, 1), edge_left_f0(2)+zoom_size);
left_cropped_0 = img_0_left(f0_left_y_range, f0_left_x_range);
imshow(left_cropped_0); hold on;
title('Frame 0 Left - Zoomed', 'FontSize', 12, 'FontWeight', 'bold');

left_local_x = edge_left_f0(1) - f0_left_x_range(1) + 1;
left_local_y = edge_left_f0(2) - f0_left_y_range(1) + 1;

plot(left_local_x, left_local_y, 'ro', 'MarkerSize', 18, 'LineWidth', 4);
plot(left_local_x, left_local_y, 'r*', 'MarkerSize', 25, 'LineWidth', 4);
line([left_local_x-20, left_local_x+20], [left_local_y, left_local_y], 'Color', 'red', 'LineWidth', 4);
line([left_local_x, left_local_x], [left_local_y-20, left_local_y+20], 'Color', 'red', 'LineWidth', 4);

text(left_local_x+25, left_local_y-25, 'KF Original', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Subplot 2: Frame 0 Right (zoomed around stereo pair)
subplot(2, 2, 2);
f0_right_x_range = max(1, edge_right_f0(1)-zoom_size):min(size(img_0_right, 2), edge_right_f0(1)+zoom_size);
f0_right_y_range = max(1, edge_right_f0(2)-zoom_size):min(size(img_0_right, 1), edge_right_f0(2)+zoom_size);
right_cropped_0 = img_0_right(f0_right_y_range, f0_right_x_range);
imshow(right_cropped_0); hold on;
title('Frame 0 Right - Zoomed', 'FontSize', 12, 'FontWeight', 'bold');

right_local_x = edge_right_f0(1) - f0_right_x_range(1) + 1;
right_local_y = edge_right_f0(2) - f0_right_y_range(1) + 1;

plot(right_local_x, right_local_y, 'bo', 'MarkerSize', 18, 'LineWidth', 4);
plot(right_local_x, right_local_y, 'b*', 'MarkerSize', 25, 'LineWidth', 4);
line([right_local_x-20, right_local_x+20], [right_local_y, right_local_y], 'Color', 'blue', 'LineWidth', 4);
line([right_local_x, right_local_x], [right_local_y-20, right_local_y+20], 'Color', 'blue', 'LineWidth', 4);

text(right_local_x+25, right_local_y-25, 'KF Stereo', 'Color', 'blue', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Subplot 3: Frame 1 Left (zoomed around candidates)
subplot(2, 2, 3);
f1_left_x_range = max(1, edges_frame1_left(1,1)-zoom_size):min(size(img_1_left, 2), edges_frame1_left(1,1)+zoom_size);
f1_left_y_range = max(1, edges_frame1_left(1,2)-zoom_size):min(size(img_1_left, 1), edges_frame1_left(1,2)+zoom_size);
left_cropped_1 = img_1_left(f1_left_y_range, f1_left_x_range);
imshow(left_cropped_1); hold on;
title('Frame 1 Left - Zoomed (Candidates)', 'FontSize', 12, 'FontWeight', 'bold');

% Plot candidate edges in zoom window
edges_in_window = [];
for i = 1:size(edges_frame1_left, 1)
    if edges_frame1_left(i,1) >= f1_left_x_range(1) && edges_frame1_left(i,1) <= f1_left_x_range(end) && ...
       edges_frame1_left(i,2) >= f1_left_y_range(1) && edges_frame1_left(i,2) <= f1_left_y_range(end)
        local_x = edges_frame1_left(i,1) - f1_left_x_range(1) + 1;
        local_y = edges_frame1_left(i,2) - f1_left_y_range(1) + 1;
        plot(local_x, local_y, 'go', 'MarkerSize', 12, 'LineWidth', 3);
        plot(local_x, local_y, 'g+', 'MarkerSize', 15, 'LineWidth', 3);
        text(local_x+5, local_y-5, sprintf('%d', i), 'Color', 'green', 'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);
        edges_in_window = [edges_in_window; edges_frame1_left(i,:)];
    end
end

text(5, 10, sprintf('%d candidates', size(edges_in_window, 1)), 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

% Subplot 4: Frame 1 Right (zoomed around projected edge)
subplot(2, 2, 4);
right_cropped_1 = img_1_right(y_range, x_range);
imshow(right_cropped_1); hold on;
title('Frame 1 Right - Zoomed (Projected)', 'FontSize', 12, 'FontWeight', 'bold');

% Show projected edge
proj_local_x = center_x - x_range(1) + 1;
proj_local_y = center_y - y_range(1) + 1;

plot(proj_local_x, proj_local_y, 'mo', 'MarkerSize', 18, 'LineWidth', 4);
plot(proj_local_x, proj_local_y, 'm*', 'MarkerSize', 25, 'LineWidth', 4);
line([proj_local_x-20, proj_local_x+20], [proj_local_y, proj_local_y], 'Color', 'magenta', 'LineWidth', 4);
line([proj_local_x, proj_local_x], [proj_local_y-20, proj_local_y+20], 'Color', 'magenta', 'LineWidth', 4);

text(proj_local_x+25, proj_local_y-25, 'EBVO Projected', 'Color', 'magenta', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);

% Show Frame 0 reference if it fits in window
if edge_right_f0(1) >= x_range(1) && edge_right_f0(1) <= x_range(end) && ...
   edge_right_f0(2) >= y_range(1) && edge_right_f0(2) <= y_range(end)
    ref_local_x = edge_right_f0(1) - x_range(1) + 1;
    ref_local_y = edge_right_f0(2) - y_range(1) + 1;
    plot(ref_local_x, ref_local_y, 'bx', 'MarkerSize', 20, 'LineWidth', 4);
    text(ref_local_x+5, ref_local_y+20, 'F0 Ref', 'Color', 'blue', 'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.9]);
end

text(5, 10, 'EBVO Output', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7]);

% Customize zoomed figure
set(gcf, 'Color', 'w');
sgtitle('Detailed View: Edge Projection and Candidate Matching', 'FontSize', 16, 'FontWeight', 'bold');

% Save the detailed figure
saveas(gcf, 'ebvo_detailed_edge_projection.png');
fprintf('Detailed EBVO edge projection view saved as ebvo_detailed_edge_projection.png\n');

% Display comprehensive analysis
fprintf('\n=== EBVO Edge Tracking Analysis ===\n');
fprintf('FRAME 0 (Keyframe):\n');
fprintf('  Left edge:    (%.2f, %.2f)\n', edge_left_f0(1), edge_left_f0(2));
fprintf('  Right edge:   (%.3f, %.3f)\n', edge_right_f0(1), edge_right_f0(2));
fprintf('  Disparity:    %.3f pixels\n', disparity_f0);
fprintf('\nFRAME 1 (Current):\n');
fprintf('  Left candidates: %d edges\n', size(edges_frame1_left, 1));
fprintf('  Right projected: (%.3f, %.3f)\n', edge_right_f1_projected(1), edge_right_f1_projected(2));
fprintf('  Motion (right):  (%.3f, %.3f)\n', motion_x, motion_y);
if ~isempty(edges_frame1_left)
    fprintf('  Est. disparity:  %.3f pixels\n', estimated_disparity_f1);
    closest_left = edges_frame1_left(1,:);
    left_distance = sqrt((closest_left(1) - edge_left_f0(1))^2 + (closest_left(2) - edge_left_f0(2))^2);
    fprintf('  Closest left:    (%.2f, %.2f) [dist: %.1f px]\n', closest_left(1), closest_left(2), left_distance);
end
fprintf('\nTRACKING SUMMARY:\n');
fprintf('  Right edge motion: %.1f pixels\n', sqrt(motion_x^2 + motion_y^2));
fprintf('  Disparity change:  %.3f pixels\n', abs(estimated_disparity_f1 - disparity_f0));
fprintf('\nFIGURES SAVED:\n');
fprintf('  1. ebvo_edge_tracking_analysis.png - Full stereo temporal view\n');
fprintf('  2. ebvo_detailed_edge_projection.png - Detailed projection analysis\n');
fprintf('=========================================\n');