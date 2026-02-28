% Visualize stereo edge matches from finalize_stereo_edge_mates output
% Creates three figures: True Positives, Inaccurate matches, and False matches

clear; close all;

% Dataset paths
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";

% For frame 0 (previous frame)
stereo_pair_0 = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
img_path_0_left = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im0.png";
img_0_left = imread(img_path_0_left);
img_path_0_right = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im1.png";
img_0_right = imread(img_path_0_right);

% Read the output file
output_file = '../output_files/frame_0_final_matches.txt';
fid = fopen(output_file, 'r');

if fid == -1
    error('Could not open file: %s\nMake sure you are running from the test/ directory or adjust the path.', output_file);
end

% Storage for edge data
tp_edges = struct('left_x', {}, 'left_y', {}, 'left_ori', {}, ...
                  'right_x', {}, 'right_y', {}, 'right_ori', {});
inaccurate_edges = struct('left_x', {}, 'left_y', {}, 'left_ori', {}, ...
                          'right_x', {}, 'right_y', {}, 'right_ori', {});
false_edges = struct('left_x', {}, 'left_y', {}, 'left_ori', {}, ...
                     'right_x', {}, 'right_y', {}, 'right_ori', {});

tp_count = 0;
inaccurate_count = 0;
false_count = 0;

% Parse the file
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    
    % Parse the line using regular expressions
    % Format: Index: N | Left Edge: Location: (x, y) Orientation: θ <--> Right Edge: Location: (x, y) Orientation: θ | is_TP: b | is_inaccurate: b | is_false: b
    
    % Extract left edge location
    left_loc = regexp(line, 'Left Edge: Location: \(([^,]+), ([^)]+)\)', 'tokens');
    if isempty(left_loc)
        continue;
    end
    left_x = str2double(left_loc{1}{1});
    left_y = str2double(left_loc{1}{2});
    
    % Extract left edge orientation
    left_ori = regexp(line, 'Left Edge:.*Orientation: ([-\d.]+)', 'tokens');
    left_orientation = str2double(left_ori{1}{1});
    
    % Extract right edge location
    right_loc = regexp(line, 'Right Edge: Location: \(([^,]+), ([^)]+)\)', 'tokens');
    right_x = str2double(right_loc{1}{1});
    right_y = str2double(right_loc{1}{2});
    
    % Extract right edge orientation
    right_ori = regexp(line, 'Right Edge:.*Orientation: ([-\d.]+)', 'tokens');
    right_orientation = str2double(right_ori{1}{1});
    
    % Extract classification flags
    is_tp_match = regexp(line, 'is_TP: (\d)', 'tokens');
    is_tp = str2double(is_tp_match{1}{1});
    
    is_inaccurate_match = regexp(line, 'is_inaccurate: (\d)', 'tokens');
    is_inaccurate = str2double(is_inaccurate_match{1}{1});
    
    is_false_match = regexp(line, 'is_false: (\d)', 'tokens');
    is_false = str2double(is_false_match{1}{1});
    
    % Store in appropriate structure
    edge_data = struct('left_x', left_x, 'left_y', left_y, 'left_ori', left_orientation, ...
                       'right_x', right_x, 'right_y', right_y, 'right_ori', right_orientation);
    
    if is_tp == 1
        tp_count = tp_count + 1;
        tp_edges(tp_count) = edge_data;
    elseif is_inaccurate == 1
        inaccurate_count = inaccurate_count + 1;
        inaccurate_edges(inaccurate_count) = edge_data;
    elseif is_false == 1
        false_count = false_count + 1;
        false_edges(false_count) = edge_data;
    end
end

fclose(fid);

fprintf('Parsed %d true positives, %d inaccurate, %d false matches\n', ...
        tp_count, inaccurate_count, false_count);

% Visualization parameters
edge_length = 15;  % Length of edge line to draw

% Create visualizations for each category


if inaccurate_count > 0
    plot_stereo_edges(img_0_left, img_0_right, inaccurate_edges, 'Inaccurate Matches', edge_length);
end

if false_count > 0
    plot_stereo_edges(img_0_left, img_0_right, false_edges, 'False Matches', edge_length);
end

fprintf('Visualization complete!\n');

% Function to plot edges on images
function plot_stereo_edges(img_left, img_right, edges, title_text, edge_length)
    fig = figure('Name', title_text, 'Position', [100, 100, 1600, 600]);
    
    % Reduce spacing between subplots
    gap = 0.02;  % Small gap between subplots
    margin_left = 0.01;
    margin_right = 0.01;
    margin_top = 0.05;
    margin_bottom = 0.05;
    
    subplot_width = (1 - margin_left - margin_right - gap) / 2;
    subplot_height = 1 - margin_top - margin_bottom;
    
    % Plot left image with edges
    subplot('Position', [margin_left, margin_bottom, subplot_width, subplot_height]);
    imshow(img_left); hold on;
    title(sprintf('Left Image - %s (%d edges)', title_text, length(edges)));
    
    % Generate random colors for each edge pair
    colors = rand(length(edges), 3);
    
    % Plot edges on left image
    for i = 1:length(edges)
        x = edges(i).left_x+1;
        y = edges(i).left_y+1;
        theta = edges(i).left_ori;
        
        % Calculate edge endpoints
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        x1 = x - dx/2;
        y1 = y - dy/2;
        x2 = x + dx/2;
        y2 = y + dy/2;
        
        % Plot edge line
        plot([x1, x2], [y1, y2], 'Color', colors(i, :), 'LineWidth', 3);
    end
    hold off;
    
    % Plot right image with edges
    subplot('Position', [margin_left + subplot_width + gap, margin_bottom, subplot_width, subplot_height]);
    imshow(img_right); hold on;
    title(sprintf('Right Image - %s (%d edges)', title_text, length(edges)));
    
    % Plot edges on right image with corresponding colors
    for i = 1:length(edges)
        x = edges(i).right_x+1;
        y = edges(i).right_y+1;
        theta = edges(i).right_ori;
        
        % Calculate edge endpoints
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        x1 = x - dx/2;
        y1 = y - dy/2;
        x2 = x + dx/2;
        y2 = y + dy/2;
        
        % Plot edge line with same color as left
        plot([x1, x2], [y1, y2], 'Color', colors(i, :), 'LineWidth', 3);
    end
    hold off;
    
    % Save figure as PDF with high resolution
    filename = strrep(lower(title_text), ' ', '_');
    pdf_path = sprintf('../outputs/stereo_matches_%s.pdf', filename);
    
    % Set paper size to match figure size for better quality
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperOrientation', 'landscape');
    set(fig, 'PaperSize', [16, 6]);
    
    print(fig, pdf_path, '-dpdf', '-r300');
    fprintf('Saved: %s\n', pdf_path);
end
