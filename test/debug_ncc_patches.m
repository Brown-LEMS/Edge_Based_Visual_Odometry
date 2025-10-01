% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/debug_ncc_patches.m
function debug_ncc_patches(frame_idx, edge_idx)
    % Debug NCC patches for a specific edge
    % Inputs:
    %   frame_idx - frame index (e.g., 1)
    %   edge_idx - edge index to debug (e.g., 8)
    
    % Set paths
    source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
    dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
    output_dir = sprintf('/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/ncc_debug_frame%d_edge%d', frame_idx, edge_idx);
    
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Define stereo pairs for frames
    frame_pairs = {
        "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png", % frame 0
        "images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png"  % frame 1
    };
    
    % Load images for previous and current frame
    prev_frame_idx = frame_idx - 1;
    img_path_prev = source_dataset_folder + dataset_sequence_path + frame_pairs{prev_frame_idx + 1} + "/im0.png";
    img_prev = double(imread(img_path_prev));
    
    img_path_curr = source_dataset_folder + dataset_sequence_path + frame_pairs{frame_idx + 1} + "/im0.png";
    img_curr = double(imread(img_path_curr));
    
    % Handle RGB images (convert to grayscale)
    if ndims(img_prev) > 2
        img_prev = img_prev(:,:,1); % Use first channel or rgb2gray
    end
    if ndims(img_curr) > 2
        img_curr = img_curr(:,:,1); % Use first channel or rgb2gray
    end
    
    % Extract edge data for the specified edge index
    csv_path = '/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/outputs/ncc_scores_gt_debug_frame_1.csv';
    extract_edge_data(csv_path, edge_idx);
    
    % Load extracted edge data
    prev_edge = importdata("prev.txt");
    candidates = importdata("ncc.txt");
    gt_edge = importdata("gt.txt");
    
    % Define constants to match C++ code
    PATCH_SIZE = 7;
    ORTHOGONAL_SHIFT_MAG = 3.0; % This should match your C++ code
    
    % Open file for statistics
    fid = fopen(fullfile(output_dir, 'patch_statistics.txt'), 'w');
    fprintf(fid, 'NCC Patch Statistics for Frame %d, Edge %d\n', frame_idx, edge_idx);
    fprintf(fid, '===============================================\n\n');
    
    % Debug the Previous Edge patches
    fprintf(fid, 'Previous Edge:\n');
    fprintf(fid, '  Location: (%.2f, %.2f)\n', prev_edge(1), prev_edge(2));
    fprintf(fid, '  Orientation: %.4f\n\n', prev_edge(3));
    
    % Get orthogonal shifted points for previous edge using the same algorithm as C++
    prev_shifted_plus = [
        prev_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(prev_edge(3)),
        prev_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(prev_edge(3)))
    ];
    prev_shifted_minus = [
        prev_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(prev_edge(3))),
        prev_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(prev_edge(3))
    ];
    
    % Initialize matrices for patches
    patch_coord_x_plus = zeros(PATCH_SIZE, PATCH_SIZE);
    patch_coord_y_plus = zeros(PATCH_SIZE, PATCH_SIZE);
    patch_coord_x_minus = zeros(PATCH_SIZE, PATCH_SIZE);
    patch_coord_y_minus = zeros(PATCH_SIZE, PATCH_SIZE);
    prev_patch_plus = zeros(PATCH_SIZE, PATCH_SIZE);
    prev_patch_minus = zeros(PATCH_SIZE, PATCH_SIZE);
    
    % Extract patches for previous edge using the same method as C++
    [prev_patch_plus, coords_x_plus, coords_y_plus] = extract_patch_on_one_edge_side(prev_shifted_plus, prev_edge(3), img_prev, PATCH_SIZE);
    [prev_patch_minus, coords_x_minus, coords_y_minus] = extract_patch_on_one_edge_side(prev_shifted_minus, prev_edge(3), img_prev, PATCH_SIZE);
    
    % Calculate statistics for previous patches
    mean_prev_plus = mean(prev_patch_plus(:));
    mean_prev_minus = mean(prev_patch_minus(:));
    var_prev_plus = var(prev_patch_plus(:));
    var_prev_minus = var(prev_patch_minus(:));
    
    fprintf(fid, '  Plus Patch - Mean: %.4f, Variance: %.4f\n', mean_prev_plus, var_prev_plus);
    fprintf(fid, '  Minus Patch - Mean: %.4f, Variance: %.4f\n\n', mean_prev_minus, var_prev_minus);
    
    % Save patches as images (enlarged for visibility)
    imwrite(uint8(normalize_for_display(prev_patch_plus, 20)), fullfile(output_dir, 'prev_patch_plus.png'));
    imwrite(uint8(normalize_for_display(prev_patch_minus, 20)), fullfile(output_dir, 'prev_patch_minus.png'));
    
    % Debug the GT Edge patches
    fprintf(fid, 'Ground Truth Edge:\n');
    fprintf(fid, '  Location: (%.2f, %.2f)\n', gt_edge(1), gt_edge(2));
    fprintf(fid, '  Orientation: %.4f\n\n', gt_edge(3));
    
    % Get orthogonal shifted points for GT edge
    gt_shifted_plus = [
        gt_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(gt_edge(3)),
        gt_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(gt_edge(3)))
    ];
    gt_shifted_minus = [
        gt_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(gt_edge(3))),
        gt_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(gt_edge(3))
    ];
    
    % Extract patches for GT edge
    [gt_patch_plus, ~, ~] = extract_patch_on_one_edge_side(gt_shifted_plus, gt_edge(3), img_curr, PATCH_SIZE);
    [gt_patch_minus, ~, ~] = extract_patch_on_one_edge_side(gt_shifted_minus, gt_edge(3), img_curr, PATCH_SIZE);
    
    % Calculate statistics for GT patches
    mean_gt_plus = mean(gt_patch_plus(:));
    mean_gt_minus = mean(gt_patch_minus(:));
    var_gt_plus = var(gt_patch_plus(:));
    var_gt_minus = var(gt_patch_minus(:));
    
    fprintf(fid, '  Plus Patch - Mean: %.4f, Variance: %.4f\n', mean_gt_plus, var_gt_plus);
    fprintf(fid, '  Minus Patch - Mean: %.4f, Variance: %.4f\n\n', mean_gt_minus, var_gt_minus);
    
    % Save patches as images (enlarged for visibility)
    imwrite(uint8(normalize_for_display(gt_patch_plus, 20)), fullfile(output_dir, 'gt_patch_plus.png'));
    imwrite(uint8(normalize_for_display(gt_patch_minus, 20)), fullfile(output_dir, 'gt_patch_minus.png'));
    
    % Calculate NCC scores between previous and GT using same method as C++
    gt_ncc_plus_plus = compute_similarity(prev_patch_plus, gt_patch_plus);
    gt_ncc_minus_minus = compute_similarity(prev_patch_minus, gt_patch_minus);
    gt_ncc_plus_minus = compute_similarity(prev_patch_plus, gt_patch_minus);
    gt_ncc_minus_plus = compute_similarity(prev_patch_minus, gt_patch_plus);
    
    fprintf(fid, 'NCC Scores between Previous and GT:\n');
    fprintf(fid, '  Plus-Plus: %.4f\n', gt_ncc_plus_plus);
    fprintf(fid, '  Minus-Minus: %.4f\n', gt_ncc_minus_minus);
    fprintf(fid, '  Plus-Minus: %.4f\n', gt_ncc_plus_minus);
    fprintf(fid, '  Minus-Plus: %.4f\n', gt_ncc_minus_plus);
    fprintf(fid, '  Max: %.4f\n\n', max([gt_ncc_plus_plus, gt_ncc_minus_minus, gt_ncc_plus_minus, gt_ncc_minus_plus]));
    
    % Process candidate edges
    fprintf(fid, 'Candidate Edges: %d\n\n', size(candidates, 1));
    
    % Limit to top 10 candidates for readability
    max_candidates = min(10, size(candidates, 1));
    
    % Create visualization of previous image with edge and patches
    figure('Position', [100, 100, 900, 800], 'Visible', 'off');
    imshow(uint8(img_prev), []); hold on;
    title(sprintf('Previous Frame Edge %d with Patches', edge_idx), 'FontSize', 14);
    
    % Plot the previous edge
    plot(prev_edge(1), prev_edge(2), 'r*', 'MarkerSize', 10);
    
    % Draw orientation vector
    quiver(prev_edge(1), prev_edge(2), cos(prev_edge(3)), sin(prev_edge(3)), 15, 'r', 'LineWidth', 2);
    
    % Draw orthogonal direction lines and patches
    plot(prev_shifted_plus(1), prev_shifted_plus(2), 'g*', 'MarkerSize', 8);
    plot(prev_shifted_minus(1), prev_shifted_minus(2), 'b*', 'MarkerSize', 8);
    
    % Draw lines to shifted points
    line([prev_edge(1), prev_shifted_plus(1)], [prev_edge(2), prev_shifted_plus(2)], 'Color', 'g', 'LineWidth', 2);
    line([prev_edge(1), prev_shifted_minus(1)], [prev_edge(2), prev_shifted_minus(2)], 'Color', 'b', 'LineWidth', 2);
    
    % Draw patch rectangles - visualize actual sampling points
    scatter(coords_x_plus(:), coords_y_plus(:), 10, 'g', 'filled');
    scatter(coords_x_minus(:), coords_y_minus(:), 10, 'b', 'filled');
    
    % Add labels
    text(prev_shifted_plus(1)+10, prev_shifted_plus(2), 'Plus Patch', 'Color', 'g', 'FontSize', 12);
    text(prev_shifted_minus(1)+10, prev_shifted_minus(2), 'Minus Patch', 'Color', 'b', 'FontSize', 12);
    
    % Save figure
    saveas(gcf, fullfile(output_dir, 'prev_edge_patches.png'));
    
    % Create visualization of current image with GT and patches
    figure('Position', [100, 100, 900, 800], 'Visible', 'off');
    imshow(uint8(img_curr), []); hold on;
    title(sprintf('Current Frame GT Edge with Patches'), 'FontSize', 14);
    
    % Plot the GT edge
    plot(gt_edge(1), gt_edge(2), 'r*', 'MarkerSize', 10);
    
    % Draw orientation vector
    quiver(gt_edge(1), gt_edge(2), cos(gt_edge(3)), sin(gt_edge(3)), 15, 'r', 'LineWidth', 2);
    
    % Draw orthogonal direction lines and patches
    plot(gt_shifted_plus(1), gt_shifted_plus(2), 'g*', 'MarkerSize', 8);
    plot(gt_shifted_minus(1), gt_shifted_minus(2), 'b*', 'MarkerSize', 8);
    
    % Draw lines to shifted points
    line([gt_edge(1), gt_shifted_plus(1)], [gt_edge(2), gt_shifted_plus(2)], 'Color', 'g', 'LineWidth', 2);
    line([gt_edge(1), gt_shifted_minus(1)], [gt_edge(2), gt_shifted_minus(2)], 'Color', 'b', 'LineWidth', 2);
    
    % Save figure
    saveas(gcf, fullfile(output_dir, 'gt_edge_patches.png'));
    
    % Create an interactive figure to explore all candidates
    figure('Position', [100, 100, 1200, 800], 'Visible', 'on');
    
    % Subplot 1: Current frame with all candidates and GT
    subplot(2, 2, 1);
    imshow(uint8(img_curr), []); hold on;
    title('Current Frame with Candidates and GT', 'FontSize', 12);
    
    % Plot GT edge
    plot(gt_edge(1), gt_edge(2), 'r*', 'MarkerSize', 10);
    text(gt_edge(1)+5, gt_edge(2), 'GT', 'Color', 'red', 'FontSize', 10);
    
    % Plot all candidates
    for i = 1:size(candidates, 1)
        % Calculate distance to GT
        dist_to_gt = norm([candidates(i,1) - gt_edge(1), candidates(i,2) - gt_edge(2)]);
        
        % Color based on distance to GT
        if dist_to_gt < 3.0
            color = 'g'; % Close to GT - green
        else
            color = 'b'; % Far from GT - blue
        end
        
        % Plot candidate
        plot(candidates(i,1), candidates(i,2), [color '.'], 'MarkerSize', 10);
        
        % Add index for first few candidates
        if i <= 5
            text(candidates(i,1)+5, candidates(i,2), num2str(i), 'Color', color, 'FontSize', 10);
        end
    end
    
    % Subplot 2: Interactive candidate selection
    subplot(2, 2, 2);
    % This will be updated when a candidate is selected
    title('Select a candidate in the left plot', 'FontSize', 12);
    
    % Subplot 3: Patch visualization for previous edge
    subplot(2, 2, 3);
    imshow(uint8(normalize_for_display(prev_patch_plus, 20)));
    title('Previous Edge Plus Patch', 'FontSize', 12);
    
    % Subplot 4: Patch visualization for selected candidate
    subplot(2, 2, 4);
    imshow(uint8(normalize_for_display(gt_patch_plus, 20)));
    title('GT Edge Plus Patch', 'FontSize', 12);
    
    % Save the figure for later interactive use
    saveas(gcf, fullfile(output_dir, 'patch_explorer.fig'));
    
    % Create table with all candidate NCC scores for comparison
    cand_data = cell(max_candidates, 5);
    
    % Process each candidate edge
    for i = 1:max_candidates
        cand_edge = candidates(i,:);
        
        % Get orthogonal shifted points for candidate
        cand_shifted_plus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(cand_edge(3)),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(cand_edge(3)))
        ];
        cand_shifted_minus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(cand_edge(3))),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(cand_edge(3))
        ];
        
        % Extract patches for candidate edge
        [cand_patch_plus, ~, ~] = extract_patch_on_one_edge_side(cand_shifted_plus, cand_edge(3), img_curr, PATCH_SIZE);
        [cand_patch_minus, ~, ~] = extract_patch_on_one_edge_side(cand_shifted_minus, cand_edge(3), img_curr, PATCH_SIZE);
        
        % Calculate NCC scores with previous edge
        cand_ncc_plus_plus = compute_similarity(prev_patch_plus, cand_patch_plus);
        cand_ncc_minus_minus = compute_similarity(prev_patch_minus, cand_patch_minus);
        cand_ncc_plus_minus = compute_similarity(prev_patch_plus, cand_patch_minus);
        cand_ncc_minus_plus = compute_similarity(prev_patch_minus, cand_patch_plus);
        cand_ncc_max = max([cand_ncc_plus_plus, cand_ncc_minus_minus, cand_ncc_plus_minus, cand_ncc_minus_plus]);
        
        % Calculate distance to GT
        dist_to_gt = norm([cand_edge(1) - gt_edge(1), cand_edge(2) - gt_edge(2)]);
        
        % Save patches
        imwrite(uint8(normalize_for_display(cand_patch_plus, 20)), ...
                fullfile(output_dir, sprintf('cand%d_patch_plus.png', i)));
        imwrite(uint8(normalize_for_display(cand_patch_minus, 20)), ...
                fullfile(output_dir, sprintf('cand%d_patch_minus.png', i)));
        
        % Log to file
        fprintf(fid, 'Candidate %d:\n', i);
        fprintf(fid, '  Location: (%.2f, %.2f)\n', cand_edge(1), cand_edge(2));
        fprintf(fid, '  Orientation: %.4f\n', cand_edge(3));
        fprintf(fid, '  Distance to GT: %.4f\n', dist_to_gt);
        fprintf(fid, '  NCC Scores with Previous:\n');
        fprintf(fid, '    Plus-Plus: %.4f\n', cand_ncc_plus_plus);
        fprintf(fid, '    Minus-Minus: %.4f\n', cand_ncc_minus_minus);
        fprintf(fid, '    Plus-Minus: %.4f\n', cand_ncc_plus_minus);
        fprintf(fid, '    Minus-Plus: %.4f\n', cand_ncc_minus_plus);
        fprintf(fid, '    Max: %.4f\n\n', cand_ncc_max);
        
        % Add to table data
        cand_data{i,1} = i;
        cand_data{i,2} = sprintf('(%.1f, %.1f)', cand_edge(1), cand_edge(2));
        cand_data{i,3} = dist_to_gt;
        cand_data{i,4} = cand_ncc_max;
        if dist_to_gt < 3.0
            cand_data{i,5} = 'Yes';
        else
            cand_data{i,5} = 'No';
        end
        
        % Create an individual visualization for this candidate
        figure('Position', [100, 100, 900, 800], 'Visible', 'off');
        imshow(uint8(img_curr), []); hold on;
        title(sprintf('Candidate %d Edge with Patches', i), 'FontSize', 14);
        
        % Plot the candidate edge
        plot(cand_edge(1), cand_edge(2), 'r*', 'MarkerSize', 10);
        
        % Draw orientation vector
        quiver(cand_edge(1), cand_edge(2), cos(cand_edge(3)), sin(cand_edge(3)), 15, 'r', 'LineWidth', 2);
        
        % Draw orthogonal direction lines and patches
        plot(cand_shifted_plus(1), cand_shifted_plus(2), 'g*', 'MarkerSize', 8);
        plot(cand_shifted_minus(1), cand_shifted_minus(2), 'b*', 'MarkerSize', 8);
        
        % Draw lines to shifted points
        line([cand_edge(1), cand_shifted_plus(1)], [cand_edge(2), cand_shifted_plus(2)], 'Color', 'g', 'LineWidth', 2);
        line([cand_edge(1), cand_shifted_minus(1)], [cand_edge(2), cand_shifted_minus(2)], 'Color', 'b', 'LineWidth', 2);
        
        % Add the GT point for reference
        plot(gt_edge(1), gt_edge(2), 'm*', 'MarkerSize', 8);
        text(gt_edge(1)+5, gt_edge(2), 'GT', 'Color', 'magenta', 'FontSize', 12);
        
        % Add NCC info
        text(20, 30, sprintf('NCC Max: %.4f', cand_ncc_max), 'Color', 'yellow', 'FontSize', 14, 'BackgroundColor', 'black');
        text(20, 60, sprintf('Distance to GT: %.2f', dist_to_gt), 'Color', 'yellow', 'FontSize', 14, 'BackgroundColor', 'black');
        
        % Save figure
        saveas(gcf, fullfile(output_dir, sprintf('candidate_%d_visual.png', i)));
    end
    
    % Save table as CSV
    table_file = fopen(fullfile(output_dir, 'candidate_scores.csv'), 'w');
    fprintf(table_file, 'Candidate,Position,Distance to GT,Max NCC,Near GT?\n');
    for i = 1:max_candidates
        fprintf(table_file, '%d,%s,%.4f,%.4f,%s\n', ...
                cand_data{i,1}, cand_data{i,2}, cand_data{i,3}, cand_data{i,4}, cand_data{i,5});
    end
    fclose(table_file);
    
    % Create a grid visualization of all patches
    figure('Position', [100, 100, 1200, 800], 'Visible', 'off');
    
    % Create subplots for all patches
    % Row 1: Previous and GT patches
    subplot(4, 5, 1);
    imshow(uint8(normalize_for_display(prev_patch_plus, 10))); title('Prev+', 'FontSize', 10);
    subplot(4, 5, 2);
    imshow(uint8(normalize_for_display(prev_patch_minus, 10))); title('Prev-', 'FontSize', 10);
    subplot(4, 5, 3);
    imshow(uint8(normalize_for_display(gt_patch_plus, 10))); title('GT+', 'FontSize', 10);
    subplot(4, 5, 4);
    imshow(uint8(normalize_for_display(gt_patch_minus, 10))); title('GT-', 'FontSize', 10);
    
    % Rows 2-4: Candidate patches
    for i = 1:min(15, size(candidates, 1))
        cand_edge = candidates(i,:);
        
        % Get orthogonal shifted points for candidate
        cand_shifted_plus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(cand_edge(3)),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(cand_edge(3)))
        ];
        
        % Extract patch for candidate edge
        [cand_patch_plus, ~, ~] = extract_patch_on_one_edge_side(cand_shifted_plus, cand_edge(3), img_curr, PATCH_SIZE);
        
        % Calculate NCC with prev
        ncc = compute_similarity(prev_patch_plus, cand_patch_plus);
        
        % Only show plus patch to save space
        row = floor((i+4)/5) + 1;
        col = mod((i+4), 5) + 1;
        subplot(4, 5, (row-1)*5 + col);
        imshow(uint8(normalize_for_display(cand_patch_plus, 10)));
        title(sprintf('C%d (%.2f)', i, ncc), 'FontSize', 8);
    end
    
    % Save grid visualization
    saveas(gcf, fullfile(output_dir, 'all_patches_grid.png'));
    % Add after line ~325 in debug_ncc_patches.m (after saving all_patches_grid.png)
    % Create detailed patch visualization for each candidate and previous edge
    % with all plus/minus patch pairs for easy comparison
    figure('Position', [100, 100, 1200, 900], 'Visible', 'off');
    
    % Create a more detailed patch visualization showing both plus and minus for each candidate
    num_candidates_to_show = min(8, size(candidates, 1));
    rows = num_candidates_to_show + 1; % +1 for previous edge patches
    cols = 4; % [plus patch, minus patch, edge visual, ncc scores]
    
    % Add previous edge patches in first row
    subplot(rows, cols, 1);
    imshow(uint8(normalize_for_display(prev_patch_plus, 15)));
    title(sprintf('Prev Edge+ (%.1f,%.1f)', prev_edge(1), prev_edge(2)), 'FontSize', 10);
    
    subplot(rows, cols, 2);
    imshow(uint8(normalize_for_display(prev_patch_minus, 15)));
    title('Prev Edge-', 'FontSize', 10);
    
    % Show previous frame with edge location
    subplot(rows, cols, 3);
    imshow(uint8(img_prev), []); hold on;
    plot(prev_edge(1), prev_edge(2), 'r*', 'MarkerSize', 8);
    plot([prev_edge(1) prev_edge(1)+15*cos(prev_edge(3))], [prev_edge(2) prev_edge(2)+15*sin(prev_edge(3))], 'r-', 'LineWidth', 2);
    plot(prev_shifted_plus(1), prev_shifted_plus(2), 'g*', 'MarkerSize', 6);
    plot(prev_shifted_minus(1), prev_shifted_minus(2), 'b*', 'MarkerSize', 6);
    title('Prev Frame Edge', 'FontSize', 10);
    
    % Empty subplot for alignment
    subplot(rows, cols, 4);
    title('Previous Frame', 'FontSize', 10);
    
    % Process candidates
    for i = 1:num_candidates_to_show
        cand_edge = candidates(i,:);
        
        % Get orthogonal shifted points for candidate
        cand_shifted_plus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(cand_edge(3)),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(cand_edge(3)))
        ];
        cand_shifted_minus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(cand_edge(3))),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(cand_edge(3))
        ];
        
        % Extract patches for candidate edge
        [cand_patch_plus, cand_coords_x_plus, cand_coords_y_plus] = extract_patch_on_one_edge_side(cand_shifted_plus, cand_edge(3), img_curr, PATCH_SIZE);
        [cand_patch_minus, cand_coords_x_minus, cand_coords_y_minus] = extract_patch_on_one_edge_side(cand_shifted_minus, cand_edge(3), img_curr, PATCH_SIZE);
        
        % Calculate NCC scores with previous edge
        cand_ncc_plus_plus = compute_similarity(prev_patch_plus, cand_patch_plus);
        cand_ncc_minus_minus = compute_similarity(prev_patch_minus, cand_patch_minus);
        cand_ncc_plus_minus = compute_similarity(prev_patch_plus, cand_patch_minus);
        cand_ncc_minus_plus = compute_similarity(prev_patch_minus, cand_patch_plus);
        cand_ncc_max = max([cand_ncc_plus_plus, cand_ncc_minus_minus, cand_ncc_plus_minus, cand_ncc_minus_plus]);
        
        % Calculate distance to GT
        dist_to_gt = norm([cand_edge(1) - gt_edge(1), cand_edge(2) - gt_edge(2)]);
        
        % Display plus patch
        subplot(rows, cols, i*cols + 1);
        imshow(uint8(normalize_for_display(cand_patch_plus, 15)));
        title(sprintf('Cand%d+ (%.1f,%.1f)', i, cand_edge(1), cand_edge(2)), 'FontSize', 10);
        
        % Display minus patch
        subplot(rows, cols, i*cols + 2);
        imshow(uint8(normalize_for_display(cand_patch_minus, 15)));
        title('Cand' + string(i) + '-', 'FontSize', 10);
        
        % Show current frame with candidate edge location
        subplot(rows, cols, i*cols + 3);
        imshow(uint8(img_curr), []); hold on;
        plot(cand_edge(1), cand_edge(2), 'r*', 'MarkerSize', 8);
        plot([cand_edge(1) cand_edge(1)+15*cos(cand_edge(3))], [cand_edge(2) cand_edge(2)+15*sin(cand_edge(3))], 'r-', 'LineWidth', 2);
        plot(cand_shifted_plus(1), cand_shifted_plus(2), 'g*', 'MarkerSize', 6);
        plot(cand_shifted_minus(1), cand_shifted_minus(2), 'b*', 'MarkerSize', 6);
        
        % Show GT point as reference
        plot(gt_edge(1), gt_edge(2), 'm*', 'MarkerSize', 6);
        title('Frame ' + string(frame_idx) + ' Cand' + string(i), 'FontSize', 10);
        
        % Display NCC scores
        subplot(rows, cols, i*cols + 4);
        axis off;
        text(0.1, 0.8, sprintf('NCC Scores:'), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.65, sprintf('Plus-Plus: %.4f', cand_ncc_plus_plus), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.5, sprintf('Minus-Minus: %.4f', cand_ncc_minus_minus), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.35, sprintf('Plus-Minus: %.4f', cand_ncc_plus_minus), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.2, sprintf('Minus-Plus: %.4f', cand_ncc_minus_plus), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.05, sprintf('Distance to GT: %.2f px', dist_to_gt), 'FontSize', 9, 'Units', 'normalized', 'Color', 'b');
        
        % Add background color based on proximity to GT
        if dist_to_gt < 3.0
            rectangle('Position',[0 0 1 1], 'FaceColor', [0.9 1 0.9], 'EdgeColor', 'none'); % light green
            text(0.5, 0.9, 'NEAR GT', 'FontSize', 10, 'HorizontalAlignment', 'center', 'Color', 'g', 'Units', 'normalized');
        end
    end
    
    % Save detailed patch grid
    saveas(gcf, fullfile(output_dir, 'detailed_patch_comparison.png'));
    
    % Create a second figure showing only the patches side by side for direct comparison
    figure('Position', [100, 100, 900, 900], 'Visible', 'off');
    
    % Number of columns in grid
    grid_cols = 5; % [prev+, prev-, cand1+, cand1-, cand2+, cand2-, etc.]
    grid_rows = num_candidates_to_show + 1; % +1 for header row
    
    % Create header
    subplot(grid_rows, grid_cols, 1);
    text(0.5, 0.5, 'Previous+', 'FontSize', 10, 'HorizontalAlignment', 'center');
    axis off;
    
    subplot(grid_rows, grid_cols, 2);
    text(0.5, 0.5, 'Previous-', 'FontSize', 10, 'HorizontalAlignment', 'center');
    axis off;
    
    subplot(grid_rows, grid_cols, 3);
    text(0.5, 0.5, 'Candidate+', 'FontSize', 10, 'HorizontalAlignment', 'center');
    axis off;
    
    subplot(grid_rows, grid_cols, 4);
    text(0.5, 0.5, 'Candidate-', 'FontSize', 10, 'HorizontalAlignment', 'center');
    axis off;
    
    subplot(grid_rows, grid_cols, 5);
    text(0.5, 0.5, 'NCC Scores', 'FontSize', 10, 'HorizontalAlignment', 'center');
    axis off;
    
    % Add previous edge patches at the top for reference
    for i = 1:num_candidates_to_show
        cand_edge = candidates(i,:);
        
        % Get orthogonal shifted points for candidate
        cand_shifted_plus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(cand_edge(3)),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(cand_edge(3)))
        ];
        cand_shifted_minus = [
            cand_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(cand_edge(3))),
            cand_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(cand_edge(3))
        ];
        
        % Extract patches for candidate edge
        [cand_patch_plus, ~, ~] = extract_patch_on_one_edge_side(cand_shifted_plus, cand_edge(3), img_curr, PATCH_SIZE);
        [cand_patch_minus, ~, ~] = extract_patch_on_one_edge_side(cand_shifted_minus, cand_edge(3), img_curr, PATCH_SIZE);
        
        % Calculate NCC scores with previous edge
        cand_ncc_plus_plus = compute_similarity(prev_patch_plus, cand_patch_plus);
        cand_ncc_minus_minus = compute_similarity(prev_patch_minus, cand_patch_minus);
        cand_ncc_plus_minus = compute_similarity(prev_patch_plus, cand_patch_minus);
        cand_ncc_minus_plus = compute_similarity(prev_patch_minus, cand_patch_plus);
        cand_ncc_max = max([cand_ncc_plus_plus, cand_ncc_minus_minus, cand_ncc_plus_minus, cand_ncc_minus_plus]);
        
        % Calculate distance to GT
        dist_to_gt = norm([cand_edge(1) - gt_edge(1), cand_edge(2) - gt_edge(2)]);
        
        % Row for this candidate
        row_idx = i + 1;
        
        % Display previous edge patches
        subplot(grid_rows, grid_cols, (row_idx-1)*grid_cols + 1);
        imshow(uint8(normalize_for_display(prev_patch_plus, 15)));
        title('Prev+', 'FontSize', 9);
        
        subplot(grid_rows, grid_cols, (row_idx-1)*grid_cols + 2);
        imshow(uint8(normalize_for_display(prev_patch_minus, 15)));
        title('Prev-', 'FontSize', 9);
        
        % Display candidate patches
        subplot(grid_rows, grid_cols, (row_idx-1)*grid_cols + 3);
        imshow(uint8(normalize_for_display(cand_patch_plus, 15)));
        title(sprintf('C%d+ (%.1f,%.1f)', i, cand_edge(1), cand_edge(2)), 'FontSize', 9);
        
        subplot(grid_rows, grid_cols, (row_idx-1)*grid_cols + 4);
        imshow(uint8(normalize_for_display(cand_patch_minus, 15)));
        title(sprintf('C%d-', i), 'FontSize', 9);
        
        % Display NCC scores
        subplot(grid_rows, grid_cols, (row_idx-1)*grid_cols + 5);
        axis off;
        
        % Add background color based on proximity to GT
        if dist_to_gt < 3.0
            rectangle('Position',[0 0 1 1], 'FaceColor', [0.9 1 0.9], 'EdgeColor', 'none'); % light green
        end
        
        % Text for NCC scores
        best_match_type = find_best_match_type([cand_ncc_plus_plus, cand_ncc_minus_minus, cand_ncc_plus_minus, cand_ncc_minus_plus]);
        text(0.1, 0.85, sprintf('Best: %.4f (%s)', cand_ncc_max, best_match_type), 'FontSize', 9, 'Units', 'normalized');
        text(0.1, 0.65, sprintf('++: %.4f', cand_ncc_plus_plus), 'FontSize', 8, 'Units', 'normalized');
        text(0.1, 0.50, sprintf('--: %.4f', cand_ncc_minus_minus), 'FontSize', 8, 'Units', 'normalized');
        text(0.1, 0.35, sprintf('+-: %.4f', cand_ncc_plus_minus), 'FontSize', 8, 'Units', 'normalized');
        text(0.1, 0.20, sprintf('-+: %.4f', cand_ncc_minus_plus), 'FontSize', 8, 'Units', 'normalized');
        text(0.1, 0.05, sprintf('D: %.2fpx', dist_to_gt), 'FontSize', 8, 'Units', 'normalized');
    end
    
    % Save side-by-side patch comparison
    saveas(gcf, fullfile(output_dir, 'side_by_side_patches.png'));
    % Close file
    fclose(fid);
    
    % Display paths to output files
    fprintf('NCC patch debug files saved to %s\n', output_dir);
    fprintf('Run "open(''%s'')" to see the interactive patch explorer\n', fullfile(output_dir, 'patch_explorer.fig'));
end

function [patch_val, coords_x, coords_y] = extract_patch_on_one_edge_side(shifted_point, theta, img, patch_size)
    % Implementation that matches the C++ get_patch_on_one_edge_side function
    % shifted_point: [x, y] center point
    % theta: orientation
    % img: input image
    % patch_size: size of the patch (should be odd)
    
    half_patch_size = floor(patch_size / 2);
    patch_val = zeros(patch_size, patch_size);
    coords_x = zeros(patch_size, patch_size);
    coords_y = zeros(patch_size, patch_size);
    
    for i = -half_patch_size:half_patch_size
        for j = -half_patch_size:half_patch_size
            % Get the rotated coordinate (same formula as C++)
            rotated_x = cos(theta)*(i) - sin(theta)*(j) + shifted_point(1);
            rotated_y = sin(theta)*(i) + cos(theta)*(j) + shifted_point(2);
            
            % Store the coordinates for visualization
            coords_x(i+half_patch_size+1, j+half_patch_size+1) = rotated_x;
            coords_y(i+half_patch_size+1, j+half_patch_size+1) = rotated_y;
            
            % Get interpolated image value
            patch_val(i+half_patch_size+1, j+half_patch_size+1) = bilinear_interpolation(img, rotated_x, rotated_y);
        end
    end
end

function val = bilinear_interpolation(img, x, y)
    % Bilinear interpolation function that mimics the C++ implementation
    
    % Get image dimensions
    [h, w] = size(img);
    
    % Check bounds
    if x < 1 || x >= w || y < 1 || y >= h
        val = 0;
        return;
    end
    
    % Get the four surrounding pixels
    x0 = floor(x);
    y0 = floor(y);
    x1 = x0 + 1;
    y1 = y0 + 1;
    
    % Ensure we don't exceed image bounds
    x1 = min(x1, w);
    y1 = min(y1, h);
    
    % Calculate interpolation weights
    wx = x - x0;
    wy = y - y0;
    
    % Perform bilinear interpolation
    val = (1-wx)*(1-wy)*img(y0,x0) + ...
          wx*(1-wy)*img(y0,x1) + ...
          (1-wx)*wy*img(y1,x0) + ...
          wx*wy*img(y1,x1);
end

function sim = compute_similarity(patch_one, patch_two)
    % Implementation that matches the C++ get_similarity function
    
    % Calculate means
    mean_one = mean(patch_one(:));
    mean_two = mean(patch_two(:));
    
    % Calculate sum of squared differences
    sum_of_squared_one = sum((patch_one(:) - mean_one).^2);
    sum_of_squared_two = sum((patch_two(:) - mean_two).^2);
    
    % Handle case with no variance
    epsilon = 1e-10;
    if sum_of_squared_one < epsilon || sum_of_squared_two < epsilon
        sim = -1.0;
        return;
    end
    
    % Calculate denominators
    denom_one = sqrt(sum_of_squared_one);
    denom_two = sqrt(sum_of_squared_two);
    
    % Normalize patches
    norm_one = (patch_one - mean_one) / denom_one;
    norm_two = (patch_two - mean_two) / denom_two;
    
    % Calculate dot product
    sim = norm_one(:)' * norm_two(:);
end
function match_type = find_best_match_type(ncc_scores)
    % Returns the type of match (++, --, +-, -+) that gave the highest NCC
    [~, max_idx] = max(ncc_scores);
    match_types = {'++', '--', '+-', '-+'};
    match_type = match_types{max_idx};
end
function img_out = normalize_for_display(img, scale_factor)
    % Normalize image to 0-255 range and optionally resize for display
    if nargin < 2
        scale_factor = 1;
    end
    
    img_norm = img - min(img(:));
    if max(img_norm(:)) > 0
        img_norm = img_norm / max(img_norm(:)) * 255;
    end
    
    if scale_factor > 1
        img_out = imresize(img_norm, scale_factor, 'nearest');
    else
        img_out = img_norm;
    end
end