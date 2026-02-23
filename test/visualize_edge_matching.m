% Interactive visualization of edge matching between left, right, and candidate edges
% Allows zoom, pan, and interactive inspection of edge correspondences
function visualize_edge_matching()
    % Set paths
    source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
    dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
    
    % Frame 1 paths
    frame_pair = "images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png";
    left_img_path = source_dataset_folder + dataset_sequence_path + frame_pair + "/im0.png";
    right_img_path = source_dataset_folder + dataset_sequence_path + frame_pair + "/im1.png";
    
    % Load images
    left_img = imread(left_img_path);
    right_img = imread(right_img_path);
    
    % Convert to grayscale if needed
    if ndims(left_img) > 2
        left_img = rgb2gray(left_img);
    end
    if ndims(right_img) > 2
        right_img = rgb2gray(right_img);
    end
    
    % Parse edge data from CSV
    % Veridical edges (LEFT image - RED)
    veridical_edges = [
        467.70, 470.09;
        467.59, 470.55;
        467.47, 470.98;
        467.14, 471.61;
        467.31, 471.38
    ];
    
    % Candidate edges (LEFT image - BLUE)
    candidate_edges = [
        469.06, 464.64; 469.46, 464.42; 468.56, 465.13; 468.48, 465.46; 468.43, 466.41;
        468.38, 467.40; 468.21, 468.12; 468.08, 468.54; 467.86, 469.44; 467.77, 469.90;
        459.87, 472.48; 459.94, 472.99; 459.97, 473.50; 459.95, 475.00; 467.70, 470.09; 467.47, 470.98
    ];
    
    % Right frame candidate edges (RIGHT image - GREEN)
    right_edges = [
        425.95, 466.00; 426.07, 466.50; 426.19, 467.00; 426.29, 467.50; 426.36, 468.00;
        426.43, 468.50; 426.49, 469.00; 426.54, 469.50; 445.48, 469.00; 445.51, 469.50;
        426.60, 470.00; 426.66, 470.51; 426.72, 471.01; 426.79, 471.50; 426.85, 472.00;
        426.91, 472.50; 427.07, 473.50; 427.16, 473.99; 427.27, 474.53; 427.38, 475.02;
        427.50, 475.50; 427.61, 475.98; 427.72, 476.44; 427.85, 477.05; 427.93, 477.53;
        427.98, 478.01; 428.01, 478.50; 428.01, 479.00; 445.57, 470.00; 445.65, 470.51;
        445.77, 470.98; 445.90, 471.49; 446.04, 472.01; 446.18, 472.52; 446.32, 472.97;
        449.65, 473.20; 446.37, 473.48; 446.38, 473.98; 446.35, 474.48; 446.32, 474.98;
        446.30, 475.49; 446.31, 475.99; 446.36, 476.50; 446.54, 477.00; 444.64, 477.52;
        444.79, 477.48; 447.44, 477.51; 444.02, 478.00; 448.13, 477.96; 443.67, 478.53;
        448.68, 478.43; 448.85, 478.57; 449.14, 478.93
    ];
    
    % Create figure with two subplots
    fig = figure('Name', 'Interactive Edge Matching Visualization', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 1600, 700]);
    
    % Left subplot: Left image with veridical (red) and candidate (blue) edges
    ax_left = subplot(1, 2, 1);
    imshow(left_img, [], 'Parent', ax_left);
    hold(ax_left, 'on');
    title(ax_left, 'Left Image - Veridical (Red) vs Candidate (Blue)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Plot veridical edges in RED
    plot(ax_left, veridical_edges(:, 1), veridical_edges(:, 2), 'ro', ...
         'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Veridical Edges (5)');
    
    % Plot candidate edges in BLUE
    plot(ax_left, candidate_edges(:, 1), candidate_edges(:, 2), 'bs', ...
         'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Candidate Edges (16)');
    
    legend(ax_left, 'Location', 'best', 'FontSize', 10);
    grid(ax_left, 'on');
    
    % Enable zoom and pan
    set(zoom(fig), 'ActionPostCallback', @updateCoords);
    set(pan(fig), 'ActionPostCallback', @updateCoords);
    
    % Right subplot: Right image with stereo candidate edges in GREEN
    ax_right = subplot(1, 2, 2);
    imshow(right_img, [], 'Parent', ax_right);
    hold(ax_right, 'on');
    title(ax_right, 'Right Image - Stereo Candidates (Green)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Plot right frame edges in GREEN
    plot(ax_right, right_edges(:, 1), right_edges(:, 2), 'g^', ...
         'MarkerSize', 7, 'LineWidth', 1.5, 'DisplayName', sprintf('Stereo Candidates (%d)', size(right_edges, 1)));
    
    legend(ax_right, 'Location', 'best', 'FontSize', 10);
    grid(ax_right, 'on');
    
    % Enable zoom and pan for right axis
    set(zoom(fig), 'ActionPostCallback', @updateCoords);
    set(pan(fig), 'ActionPostCallback', @updateCoords);
    
    % Add interactivity: Click on edges to show details
    set(ax_left, 'ButtonDownFcn', @(src, evt) showEdgeDetails(src, evt, veridical_edges, candidate_edges, 'LEFT'));
    set(ax_right, 'ButtonDownFcn', @(src, evt) showEdgeDetails(src, evt, right_edges, [], 'RIGHT'));
    
    % Add text box for showing statistics
    annotation(fig, 'textbox', [0.02, 0.02, 0.3, 0.15], ...
               'String', sprintf(['Left Image:\n' ...
                                  'Veridical: %d edges (RED)\n' ...
                                  'Candidates: %d edges (BLUE)\n\n' ...
                                  'Right Image:\n' ...
                                  'Stereo Candidates: %d edges (GREEN)'], ...
                                 size(veridical_edges, 1), size(candidate_edges, 1), size(right_edges, 1)), ...
               'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 10);
    
    % Keyboard shortcuts info
    annotation(fig, 'textbox', [0.65, 0.02, 0.3, 0.15], ...
               'String', ['Keyboard Shortcuts:\n' ...
                          'Z: Zoom mode\n' ...
                          'P: Pan mode\n' ...
                          'R: Reset view\n' ...
                          'Click on edges for details'], ...
               'BackgroundColor', 'lightyellow', 'EdgeColor', 'gray', 'FontSize', 9);
    
    % Store data in figure for access in callbacks
    fig.UserData.ax_left = ax_left;
    fig.UserData.ax_right = ax_right;
    fig.UserData.veridical_edges = veridical_edges;
    fig.UserData.candidate_edges = candidate_edges;
    fig.UserData.right_edges = right_edges;
    
    % Keyboard callback for shortcuts
    set(fig, 'KeyPressFcn', @(src, evt) handleKeyPress(src, evt, fig));
    
    function handleKeyPress(src, evt, fig)
        switch evt.Key
            case 'z'
                zoom(fig, 'on');
            case 'p'
                pan(fig, 'on');
            case 'r'
                % Reset view
                axis(fig.UserData.ax_left, 'auto');
                axis(fig.UserData.ax_right, 'auto');
            case 'escape'
                zoom(fig, 'off');
                pan(fig, 'off');
        end
    end
    
    function showEdgeDetails(ax, evt, primary_edges, secondary_edges, side)
        click_pt = get(ax, 'CurrentPoint');
        x_click = click_pt(1, 1);
        y_click = click_pt(1, 2);
        
        % Find closest edge to click
        if strcmp(side, 'LEFT')
            dist_primary = sqrt(sum((primary_edges - [x_click, y_click]).^2, 2));
            [~, idx_primary] = min(dist_primary);
            
            if ~isempty(secondary_edges)
                dist_secondary = sqrt(sum((secondary_edges - [x_click, y_click]).^2, 2));
                [~, idx_secondary] = min(dist_secondary);
                
                fprintf('\n--- LEFT IMAGE DETAILS ---\n');
                fprintf('Closest VERIDICAL edge: (%.2f, %.2f) - Distance: %.2f px\n', ...
                        primary_edges(idx_primary, 1), primary_edges(idx_primary, 2), dist_primary(idx_primary));
                fprintf('Closest CANDIDATE edge: (%.2f, %.2f) - Distance: %.2f px\n', ...
                        secondary_edges(idx_secondary, 1), secondary_edges(idx_secondary, 2), dist_secondary(idx_secondary));
            end
        else
            dist = sqrt(sum((primary_edges - [x_click, y_click]).^2, 2));
            [~, idx] = min(dist);
            
            fprintf('\n--- RIGHT IMAGE DETAILS ---\n');
            fprintf('Closest STEREO edge: (%.2f, %.2f) - Distance: %.2f px\n', ...
                    primary_edges(idx, 1), primary_edges(idx, 2), dist(idx));
        end
    end
    
    function updateCoords(~, ~)
        % Placeholder for zoom/pan updates
    end
end
