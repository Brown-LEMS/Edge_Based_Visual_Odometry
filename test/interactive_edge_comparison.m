% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/interactive_edge_comparison.m
function interactive_edge_comparison(frame_idx, edge_idx)
    % Interactive detailed edge patch comparison
    % Inputs:
    %   frame_idx - frame index (e.g., 1)
    %   edge_idx - edge index to debug (e.g., 8)
    
    % Set paths
    source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
    dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
    output_dir = sprintf('/oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/match_debug_frame%d_edge%d', frame_idx, edge_idx);
    
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
    img_prev_raw = imread(img_path_prev);
    % Ensure image is grayscale
    if ndims(img_prev_raw) > 2
        img_prev = double(rgb2gray(img_prev_raw));
    else
        img_prev = double(img_prev_raw);
    end
    
    img_path_curr = source_dataset_folder + dataset_sequence_path + frame_pairs{frame_idx + 1} + "/im0.png";
    img_curr_raw = imread(img_path_curr);
    % Ensure image is grayscale
    if ndims(img_curr_raw) > 2
        img_curr = double(rgb2gray(img_curr_raw));
    else
        img_curr = double(img_curr_raw);
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
    
    % Get orthogonal shifted points for previous edge using the same algorithm as C++
    prev_shifted_plus = [
        prev_edge(1) + ORTHOGONAL_SHIFT_MAG * sin(prev_edge(3)),
        prev_edge(2) + ORTHOGONAL_SHIFT_MAG * (-cos(prev_edge(3)))
    ];
    prev_shifted_minus = [
        prev_edge(1) + ORTHOGONAL_SHIFT_MAG * (-sin(prev_edge(3))),
        prev_edge(2) + ORTHOGONAL_SHIFT_MAG * cos(prev_edge(3))
    ];
    
    % Extract patches for previous edge using our custom function
    [prev_patch_plus, coords_x_plus, coords_y_plus] = extract_patch_on_one_edge_side(prev_shifted_plus, prev_edge(3), img_prev, PATCH_SIZE);
    [prev_patch_minus, coords_x_minus, coords_y_minus] = extract_patch_on_one_edge_side(prev_shifted_minus, prev_edge(3), img_prev, PATCH_SIZE);
    
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
    [gt_patch_plus, gt_coords_x_plus, gt_coords_y_plus] = extract_patch_on_one_edge_side(gt_shifted_plus, gt_edge(3), img_curr, PATCH_SIZE);
    [gt_patch_minus, gt_coords_x_minus, gt_coords_y_minus] = extract_patch_on_one_edge_side(gt_shifted_minus, gt_edge(3), img_curr, PATCH_SIZE);
    
    % Pre-compute all candidate patches and metrics
    num_candidates = size(candidates, 1);
    cand_patches_plus = cell(num_candidates, 1);
    cand_patches_minus = cell(num_candidates, 1);
    cand_coords_plus_x = cell(num_candidates, 1);
    cand_coords_plus_y = cell(num_candidates, 1);
    cand_coords_minus_x = cell(num_candidates, 1);
    cand_coords_minus_y = cell(num_candidates, 1);
    cand_ncc_pp = zeros(num_candidates, 1);
    cand_ncc_nn = zeros(num_candidates, 1);
    cand_ncc_pn = zeros(num_candidates, 1);
    cand_ncc_np = zeros(num_candidates, 1);
    cand_ncc_max = zeros(num_candidates, 1);
    cand_dist_to_gt = zeros(num_candidates, 1);
    cand_shifted_plus_pts = zeros(num_candidates, 2);
    cand_shifted_minus_pts = zeros(num_candidates, 2);
    
    for i = 1:num_candidates
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
        
        % Store shifted points
        cand_shifted_plus_pts(i,:) = cand_shifted_plus;
        cand_shifted_minus_pts(i,:) = cand_shifted_minus;
        
        % Extract patches for candidate edge
        [cand_patch_plus, coords_plus_x, coords_plus_y] = extract_patch_on_one_edge_side(cand_shifted_plus, cand_edge(3), img_curr, PATCH_SIZE);
        [cand_patch_minus, coords_minus_x, coords_minus_y] = extract_patch_on_one_edge_side(cand_shifted_minus, cand_edge(3), img_curr, PATCH_SIZE);
        
        % Store patches and coordinates
        cand_patches_plus{i} = cand_patch_plus;
        cand_patches_minus{i} = cand_patch_minus;
        cand_coords_plus_x{i} = coords_plus_x;
        cand_coords_plus_y{i} = coords_plus_y;
        cand_coords_minus_x{i} = coords_minus_x;
        cand_coords_minus_y{i} = coords_minus_y;
        
        % Calculate NCC scores with previous edge
        cand_ncc_pp(i) = compute_similarity(prev_patch_plus, cand_patch_plus);
        cand_ncc_nn(i) = compute_similarity(prev_patch_minus, cand_patch_minus);
        cand_ncc_pn(i) = compute_similarity(prev_patch_plus, cand_patch_minus);
        cand_ncc_np(i) = compute_similarity(prev_patch_minus, cand_patch_plus);
        cand_ncc_max(i) = max([cand_ncc_pp(i), cand_ncc_nn(i), cand_ncc_pn(i), cand_ncc_np(i)]);
        
        % Calculate distance to GT
        cand_dist_to_gt(i) = norm([cand_edge(1) - gt_edge(1), cand_edge(2) - gt_edge(2)]);
    end
    
    % Create figure for interactive visualization
    fig = figure('Position', [100, 100, 1400, 900], 'Name', 'Interactive Edge Comparison');
    
    % Left panel: Images with edges
    img_panel = uipanel('Parent', fig, 'Position', [0.01, 0.01, 0.55, 0.98]);
    
    % Right panel: Patch comparison
    patch_panel = uipanel('Parent', fig, 'Position', [0.57, 0.01, 0.42, 0.98]);
    
    % Previous frame with edge
    prev_ax = axes('Parent', img_panel, 'Position', [0.05, 0.55, 0.9, 0.4]);
    imshow(uint8(img_prev), [], 'Parent', prev_ax);
    title(prev_ax, 'Previous Frame Edge');
    hold(prev_ax, 'on');
    
    % Plot the previous edge
    plot(prev_ax, prev_edge(1), prev_edge(2), 'r*', 'MarkerSize', 10);
    
    % Draw orientation vector
    quiver(prev_ax, prev_edge(1), prev_edge(2), cos(prev_edge(3)), sin(prev_edge(3)), 15, 'r', 'LineWidth', 2);
    
    % Draw orthogonal direction lines and patches
    plot(prev_ax, prev_shifted_plus(1), prev_shifted_plus(2), 'g*', 'MarkerSize', 8);
    plot(prev_ax, prev_shifted_minus(1), prev_shifted_minus(2), 'b*', 'MarkerSize', 8);
    
    % Draw lines to shifted points
    line(prev_ax, [prev_edge(1), prev_shifted_plus(1)], [prev_edge(2), prev_shifted_plus(2)], 'Color', 'g', 'LineWidth', 2);
    line(prev_ax, [prev_edge(1), prev_shifted_minus(1)], [prev_edge(2), prev_shifted_minus(2)], 'Color', 'b', 'LineWidth', 2);
    
    % Draw patch sampling points
    scatter(prev_ax, coords_x_plus(:), coords_y_plus(:), 10, 'g', 'filled');
    scatter(prev_ax, coords_x_minus(:), coords_y_minus(:), 10, 'b', 'filled');
    
    % Add labels
    text(prev_ax, prev_shifted_plus(1)+10, prev_shifted_plus(2), 'Plus Patch', 'Color', 'g', 'FontSize', 12);
    text(prev_ax, prev_shifted_minus(1)+10, prev_shifted_minus(2), 'Minus Patch', 'Color', 'b', 'FontSize', 12);
    
    % Current frame with candidates and GT
    curr_ax = axes('Parent', img_panel, 'Position', [0.05, 0.05, 0.9, 0.4]);
    imshow(uint8(img_curr), [], 'Parent', curr_ax);
    title(curr_ax, 'Current Frame - Click on a candidate to compare');
    hold(curr_ax, 'on');
    
    % Plot GT edge
    h_gt = plot(curr_ax, gt_edge(1), gt_edge(2), 'r*', 'MarkerSize', 10);
    text(curr_ax, gt_edge(1)+5, gt_edge(2), 'GT', 'Color', 'red', 'FontSize', 12);
    
    % Plot all candidates
    h_candidates = gobjects(num_candidates, 1);
    h_labels = gobjects(num_candidates, 1);
    for i = 1:num_candidates
        cand_edge = candidates(i,:);
        
        % Color based on distance to GT
        if cand_dist_to_gt(i) < 3.0
            color = 'g'; % Close to GT - green
        else
            color = 'b'; % Far from GT - blue
        end
        
        % Plot candidate
        h_candidates(i) = plot(curr_ax, cand_edge(1), cand_edge(2), [color '.'], 'MarkerSize', 12);
        
        % Add index for all candidates
        h_labels(i) = text(curr_ax, cand_edge(1)+5, cand_edge(2), num2str(i), 'Color', color, 'FontSize', 10);
    end
    
    % Create patch comparison axes
    % Previous patches
    prev_plus_ax = axes('Parent', patch_panel, 'Position', [0.1, 0.75, 0.35, 0.2]);
    imshow(uint8(normalize_for_display(prev_patch_plus, 30)), [], 'Parent', prev_plus_ax);
    title(prev_plus_ax, 'Previous Plus Patch');
    
    prev_minus_ax = axes('Parent', patch_panel, 'Position', [0.55, 0.75, 0.35, 0.2]);
    imshow(uint8(normalize_for_display(prev_patch_minus, 30)), [], 'Parent', prev_minus_ax);
    title(prev_minus_ax, 'Previous Minus Patch');
    
    % Candidate patches
    cand_plus_ax = axes('Parent', patch_panel, 'Position', [0.1, 0.45, 0.35, 0.2]);
    h_cand_plus = imshow(zeros(PATCH_SIZE*30), [], 'Parent', cand_plus_ax);
    title(cand_plus_ax, 'Candidate Plus Patch');
    
    cand_minus_ax = axes('Parent', patch_panel, 'Position', [0.55, 0.45, 0.35, 0.2]);
    h_cand_minus = imshow(zeros(PATCH_SIZE*30), [], 'Parent', cand_minus_ax);
    title(cand_minus_ax, 'Candidate Minus Patch');
    
    % NCC scores display
    ncc_ax = axes('Parent', patch_panel, 'Position', [0.1, 0.1, 0.8, 0.25]);
    axis(ncc_ax, 'off');
    title(ncc_ax, 'NCC Scores');
    
    h_ncc_text = text(ncc_ax, 0.1, 0.9, 'Click on a candidate to view NCC scores', 'FontSize', 8);
    
    % GT patches for reference
    gt_plus_ax = axes('Parent', patch_panel, 'Position', [0.1, 0.25, 0.35, 0.15]);
    imshow(uint8(normalize_for_display(gt_patch_plus, 30)), [], 'Parent', gt_plus_ax);
    title(gt_plus_ax, 'GT Plus Patch');
    
    gt_minus_ax = axes('Parent', patch_panel, 'Position', [0.55, 0.25, 0.35, 0.15]);
    imshow(uint8(normalize_for_display(gt_patch_minus, 30)), [], 'Parent', gt_minus_ax);
    title(gt_minus_ax, 'GT Minus Patch');
    
    % Click callback function for selecting candidates
    function selectCandidate(~, event)
        % Get coordinates of the click
        pt = event.IntersectionPoint(1:2);
        
        % Find the closest candidate
        dists = sqrt(sum((candidates(:,1:2) - repmat(pt, num_candidates, 1)).^2, 2));
        [~, idx] = min(dists);
        
        % Update visualization for selected candidate
        cand_edge = candidates(idx,:);
        
        % Update current frame plot - highlight selected candidate
        for i = 1:num_candidates
            if i == idx
                h_candidates(i).MarkerSize = 15;
                h_candidates(i).LineWidth = 2;
                h_labels(i).FontSize = 14;
                h_labels(i).FontWeight = 'bold';
            else
                h_candidates(i).MarkerSize = 12;
                h_candidates(i).LineWidth = 0.5;
                h_labels(i).FontSize = 10;
                h_labels(i).FontWeight = 'normal';
            end
        end
        
        % Draw candidate patches
        cand_plus_ax.Title.String = sprintf('Candidate %d Plus Patch', idx);
        set(h_cand_plus, 'CData', uint8(normalize_for_display(cand_patches_plus{idx}, 30)));
        
        cand_minus_ax.Title.String = sprintf('Candidate %d Minus Patch', idx);
        set(h_cand_minus, 'CData', uint8(normalize_for_display(cand_patches_minus{idx}, 30)));
        
        % Update NCC scores display
        best_match_type = find_best_match_type([cand_ncc_pp(idx), cand_ncc_nn(idx), cand_ncc_pn(idx), cand_ncc_np(idx)]);
        score_text = sprintf('Candidate %d NCC Scores:\n', idx);
        score_text = [score_text, sprintf('  Plus-Plus: %.4f\n', cand_ncc_pp(idx))];
        score_text = [score_text, sprintf('  Minus-Minus: %.4f\n', cand_ncc_nn(idx))];
        score_text = [score_text, sprintf('  Plus-Minus: %.4f\n', cand_ncc_pn(idx))];
        score_text = [score_text, sprintf('  Minus-Plus: %.4f\n', cand_ncc_np(idx))];
        score_text = [score_text, sprintf('  Max: %.4f (%s)\n', cand_ncc_max(idx), best_match_type)];
        score_text = [score_text, sprintf('  Distance to GT: %.2f pixels\n', cand_dist_to_gt(idx))];
        if cand_dist_to_gt(idx) < 3.0
            score_text = [score_text, sprintf('  NEAR GROUND TRUTH')];
        end
        
        % Update the text
        set(h_ncc_text, 'String', score_text);
        
        % Add patch visualization to current frame
        cla(curr_ax);
        imshow(uint8(img_curr), [], 'Parent', curr_ax);
        hold(curr_ax, 'on');
        
        % Plot GT edge
        plot(curr_ax, gt_edge(1), gt_edge(2), 'r*', 'MarkerSize', 10);
        text(curr_ax, gt_edge(1)+5, gt_edge(2), 'GT', 'Color', 'red', 'FontSize', 12);
        
        % Plot all candidates again
        for i = 1:num_candidates
            % Color based on distance to GT
            if cand_dist_to_gt(i) < 3.0
                color = 'g'; % Close to GT - green
            else
                color = 'b'; % Far from GT - blue
            end
            
            % Plot candidate
            h_candidates(i) = plot(curr_ax, candidates(i,1), candidates(i,2), [color '.'], 'MarkerSize', 12);
            
            % Add index for all candidates
            h_labels(i) = text(curr_ax, candidates(i,1)+5, candidates(i,2), num2str(i), 'Color', color, 'FontSize', 10);
        end
        
        % Highlight selected candidate
        h_candidates(idx).MarkerSize = 15;
        h_candidates(idx).LineWidth = 2;
        h_labels(idx).FontSize = 14;
        h_labels(idx).FontWeight = 'bold';
        
        % Plot selected candidate with patches
        plot(curr_ax, cand_edge(1), cand_edge(2), 'r*', 'MarkerSize', 10);
        quiver(curr_ax, cand_edge(1), cand_edge(2), cos(cand_edge(3)), sin(cand_edge(3)), 15, 'r', 'LineWidth', 2);
        
        % Draw orthogonal direction lines and patches
        plot(curr_ax, cand_shifted_plus_pts(idx,1), cand_shifted_plus_pts(idx,2), 'g*', 'MarkerSize', 8);
        plot(curr_ax, cand_shifted_minus_pts(idx,1), cand_shifted_minus_pts(idx,2), 'b*', 'MarkerSize', 8);
        
        % Draw lines to shifted points
        line(curr_ax, [cand_edge(1), cand_shifted_plus_pts(idx,1)], [cand_edge(2), cand_shifted_plus_pts(idx,2)], 'Color', 'g', 'LineWidth', 2);
        line(curr_ax, [cand_edge(1), cand_shifted_minus_pts(idx,1)], [cand_edge(2), cand_shifted_minus_pts(idx,2)], 'Color', 'b', 'LineWidth', 2);
        
        % Draw patch sampling points
        scatter(curr_ax, cand_coords_plus_x{idx}(:), cand_coords_plus_y{idx}(:), 10, 'g', 'filled');
        scatter(curr_ax, cand_coords_minus_x{idx}(:), cand_coords_minus_y{idx}(:), 10, 'b', 'filled');
        
        % Add labels
        text(curr_ax, cand_shifted_plus_pts(idx,1)+10, cand_shifted_plus_pts(idx,2), 'Plus Patch', 'Color', 'g', 'FontSize', 12);
        text(curr_ax, cand_shifted_minus_pts(idx,1)+10, cand_shifted_minus_pts(idx,2), 'Minus Patch', 'Color', 'b', 'FontSize', 12);
        
        % Update title
        title(curr_ax, sprintf('Current Frame - Selected Candidate %d', idx));
    end

    % Set the callback for current frame axes
    set(curr_ax, 'ButtonDownFcn', @selectCandidate);
    
    % Create a dropdown menu to jump to specific candidates
    candidate_list = arrayfun(@num2str, 1:num_candidates, 'UniformOutput', false);
    uicontrol('Style', 'popupmenu', 'String', ['Select Candidate', candidate_list], ...
        'Position', [20, 20, 150, 25], 'Callback', @dropdownCallback);
    
    function dropdownCallback(src, ~)
        val = get(src, 'Value');
        if val > 1  % Skip the "Select Candidate" entry
            idx = val - 1;
            
            % Create a fake event to simulate a click
            event.IntersectionPoint = [candidates(idx,1), candidates(idx,2), 0];
            selectCandidate([], event);
        end
    end
    
    % Add option to sort candidates by NCC score or distance to GT
    uicontrol('Style', 'pushbutton', 'String', 'Sort by Max NCC', ...
        'Position', [190, 20, 150, 25], 'Callback', @sortByNCC);
    
    uicontrol('Style', 'pushbutton', 'String', 'Sort by Distance to GT', ...
        'Position', [360, 20, 150, 25], 'Callback', @sortByDistance);
    
    function sortByNCC(~, ~)
        [~, sort_idx] = sort(cand_ncc_max, 'descend');
        updateCandidateList(sort_idx);
    end
    
    function sortByDistance(~, ~)
        [~, sort_idx] = sort(cand_dist_to_gt, 'ascend');
        updateCandidateList(sort_idx);
    end
    
    function updateCandidateList(sort_idx)
        % Create sorted candidate list
        sorted_list = arrayfun(@(i) sprintf('%d (NCC:%.2f, Dist:%.1f)', i, cand_ncc_max(i), cand_dist_to_gt(i)), ...
            sort_idx, 'UniformOutput', false);
        
        % Update dropdown
        cand_dropdown = findobj('Style', 'popupmenu');
        set(cand_dropdown, 'String', ['Select Candidate', sorted_list], 'UserData', sort_idx);
        
        % Set callback to handle the sorted indices
        set(cand_dropdown, 'Callback', @sortedDropdownCallback);
    end
    
    function sortedDropdownCallback(src, ~)
        val = get(src, 'Value');
        if val > 1  % Skip the "Select Candidate" entry
            sort_idx = get(src, 'UserData');
            idx = sort_idx(val - 1);
            
            % Create a fake event to simulate a click
            event.IntersectionPoint = [candidates(idx,1), candidates(idx,2), 0];
            selectCandidate([], event);
        end
    end
    
    % Save figure
    saveas(fig, fullfile(output_dir, 'interactive_patch_explorer.fig'));
    fprintf('Interactive edge comparison saved to %s\n', fullfile(output_dir, 'interactive_patch_explorer.fig'));
    fprintf('Run "openfig(''%s'')" to open the interactive explorer\n', fullfile(output_dir, 'interactive_patch_explorer.fig'));
end

function [patch_val, coords_x, coords_y] = extract_patch_on_one_edge_side(shifted_point, theta, img, patch_size)
    
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
    % Bilinear interpolation function that exactly matches the C++ implementation
    
    % Get image dimensions
    [h, w] = size(img);
    
    % Check bounds
    if x < 1 || x >= w || y < 1 || y >= h
        val = 0;
        return;
    end
    
    % Get the four surrounding pixels (using the same naming as C++)
    Q12_x = floor(x);
    Q12_y = floor(y);
    Q22_x = ceil(x);
    Q22_y = floor(y);
    Q11_x = floor(x);
    Q11_y = ceil(y);
    Q21_x = ceil(x);
    Q21_y = ceil(y);
    
    % Ensure we don't exceed image bounds
    Q22_x = min(Q22_x, w);
    Q21_x = min(Q21_x, w);
    Q11_y = min(Q11_y, h);
    Q21_y = min(Q21_y, h);
    
    % Check for division by zero
    if Q21_x == Q11_x || Q12_y == Q11_y
        % If we can't interpolate, return nearest neighbor
        val = img(round(y), round(x));
        return;
    end
    
    % Perform bilinear interpolation exactly as in C++
    f_x_y1 = ((Q21_x - x) / (Q21_x - Q11_x)) * img(Q11_y, Q11_x) + ...
             ((x - Q11_x) / (Q21_x - Q11_x)) * img(Q21_y, Q21_x);
             
    f_x_y2 = ((Q21_x - x) / (Q21_x - Q11_x)) * img(Q12_y, Q12_x) + ...
             ((x - Q11_x) / (Q21_x - Q11_x)) * img(Q22_y, Q22_x);
             
    val = ((Q12_y - y) / (Q12_y - Q11_y)) * f_x_y1 + ...
          ((y - Q11_y) / (Q12_y - Q11_y)) * f_x_y2;
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

function match_type = find_best_match_type(ncc_scores)
    % Returns the type of match (++, --, +-, -+) that gave the highest NCC
    [~, max_idx] = max(ncc_scores);
    match_types = {'++', '--', '+-', '-+'};
    match_type = match_types{max_idx};
end

% Utility: mark specific edges on left/right images and save annotated outputs
% Usage:
%   mark_edges_on_images('/path/to/left.png', '/path/to/right.png');
%   mark_edges_on_images('/path/to/left.png', '/path/to/right.png', 'outputs');
function mark_edges_on_images(leftImagePath, rightImagePath, outDir)
    if nargin < 3
        outDir = 'outputs';
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Requested edge coordinates
    leftPt  = [136.00, 465.51];
    rightPt = [900.01, 463.92];

    % Load images
    leftImg  = imread(leftImagePath);
    rightImg = imread(rightImagePath);

    % Annotate left image
    f1 = figure('Visible', 'off');
    imshow(leftImg);
    hold on;
    plot(leftPt(1), leftPt(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(leftPt(1)+5, leftPt(2)-10, sprintf('(%.2f, %.2f)', leftPt(1), leftPt(2)), ...
        'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
    title(sprintf('Left edge at (%.2f, %.2f)', leftPt(1), leftPt(2)));
    saveas(f1, fullfile(outDir, 'left_marked.png'));
    close(f1);

    % Annotate right image
    f2 = figure('Visible', 'off');
    imshow(rightImg);
    hold on;
    plot(rightPt(1), rightPt(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    text(rightPt(1)+5, rightPt(2)-10, sprintf('(%.2f, %.2f)', rightPt(1), rightPt(2)), ...
        'Color', 'green', 'FontSize', 10, 'FontWeight', 'bold');
    title(sprintf('Right edge at (%.2f, %.2f)', rightPt(1), rightPt(2)));
    saveas(f2, fullfile(outDir, 'right_marked.png'));
    close(f2);

    fprintf('Saved annotated images to %s\n', outDir);
end