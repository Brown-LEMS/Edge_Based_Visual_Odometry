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
edge_length = 5;  % Length of edge line to draw

% Create zoomed-in visualizations for true positives
if tp_count > 0
    % Define three crop regions: [x, y, width, height]
    crop_regions = [
        413.9, 232.7, 82.8, 256.3;
        515.1, 0, 130.9, 226;
        719.5, 87.8, 222.5, 164.8
    ];
    
    % Only plot region 2
    region_idx = 2;
    plot_zoomed_region(img_0_left, img_0_right, tp_edges, ...
                      crop_regions(region_idx, :), region_idx, edge_length);
end

fprintf('Visualization complete!\n');

% Function to plot edges on images
% Function to plot edges on images
function plot_stereo_edges(img_left, img_right, edges, title_text, edge_length)
    % 1. Get image dimensions
    [img_h, img_w, ~] = size(img_left);
    
    % 2. Calculate perfect figure dimensions to eliminate all padding
    target_height = 600; 
    % We need width for 2 images side-by-side
    target_width = round(target_height * (2 * img_w) / img_h); 
    
    fig = figure('Name', title_text, 'Position', [100, 100, target_width, target_height]);
    
    % Use tiledlayout to eliminate gaps
    t = tiledlayout(1, 2, 'TileSpacing', 'none', 'Padding', 'none');
    
    % Generate random colors for each edge pair
    colors = rand(length(edges), 3);
    
    % Plot left image with edges
    nexttile;
    imshow(img_left); hold on;
    title(sprintf('Left Image - %s (%d edges)', title_text, length(edges)));
    
    % Plot edges on left image
    for i = 1:length(edges)
        x = edges(i).left_x+1;
        y = edges(i).left_y+1;
        theta = edges(i).left_ori;
        
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        plot([x - dx/2, x + dx/2], [y - dy/2, y + dy/2], 'Color', colors(i, :), 'LineWidth', 4);
    end
    hold off;
    
    % Plot right image with edges
    nexttile;
    imshow(img_right); hold on;
    title(sprintf('Right Image - %s (%d edges)', title_text, length(edges)));
    
    % Plot edges on right image
    for i = 1:length(edges)
        x = edges(i).right_x+1;
        y = edges(i).right_y+1;
        theta = edges(i).right_ori;
        
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        plot([x - dx/2, x + dx/2], [y - dy/2, y + dy/2], 'Color', colors(i, :), 'LineWidth', 4);
    end
    hold off;
    
    % Save figure as PDF with exact dimensions
    filename = strrep(lower(title_text), ' ', '_');
    pdf_path = sprintf('../outputs/stereo_matches_%s.pdf', filename);
    
    % Convert pixels to inches for PDF output (assuming standard 100 DPI)
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0, 0, target_width/100, target_height/100]);
    set(fig, 'PaperSize', [target_width/100, target_height/100]);
    
    print(fig, pdf_path, '-dpdf', '-r300');
    fprintf('Saved: %s\n', pdf_path);
end

% Function to plot zoomed-in region
function plot_zoomed_region(img_left, img_right, edges, crop_rect, region_idx, edge_length)
    % crop_rect = [x, y, width, height]
    x_start = crop_rect(1);
    y_start = crop_rect(2);
    width = crop_rect(3);
    height = crop_rect(4);
    
    % Convert to 1-based indexing and ensure integer values
    x_start_idx = max(1, round(x_start + 1));
    y_start_idx = max(1, round(y_start + 1));
    x_end_idx = min(size(img_left, 2), round(x_start + width + 1));
    y_end_idx = min(size(img_left, 1), round(y_start + height + 1));
    
    % Crop images
    img_left_crop = img_left(y_start_idx:y_end_idx, x_start_idx:x_end_idx, :);
    img_right_crop = img_right(y_start_idx:y_end_idx, x_start_idx:x_end_idx, :);
    
    % Filter edges within the crop region (using 0-based coordinates)
    edges_in_region = [];
    count = 0;
    for i = 1:length(edges)
        left_x = edges(i).left_x;
        left_y = edges(i).left_y;
        
        % Check if left edge is within crop region
        if left_x >= x_start && left_x <= (x_start + width) && ...
           left_y >= y_start && left_y <= (y_start + height)
            count = count + 1;
            edges_in_region(count).left_x = left_x - x_start;
            edges_in_region(count).left_y = left_y - y_start;
            edges_in_region(count).left_ori = edges(i).left_ori;
            edges_in_region(count).right_x = edges(i).right_x - x_start;
            edges_in_region(count).right_y = edges(i).right_y - y_start;
            edges_in_region(count).right_ori = edges(i).right_ori;
        end
    end
    
    if isempty(edges_in_region)
        fprintf('Region %d: No edges found in this region\n', region_idx);
        return;
    end
    
    fprintf('Region %d: Found %d edges\n', region_idx, length(edges_in_region));
    
    % 1. Get cropped image dimensions
    [crop_h, crop_w, ~] = size(img_left_crop);
    
    % 2. Calculate perfect figure dimensions for zoomed images
    target_height = 600;
    target_width = round(target_height * (2 * crop_w) / crop_h);
    
    % Create figure with dynamic size
    fig = figure('Name', sprintf('True Positives - Zoomed Region %d', region_idx), ...
                 'Position', [100 + region_idx*50, 100 + region_idx*50, target_width, target_height]);
    
    % Use tiledlayout for the zoomed region as well
    t = tiledlayout(1, 2, 'TileSpacing', 'none', 'Padding', 'none');
    
    % Generate random colors for each edge pair
    colors = rand(length(edges_in_region), 3);
    
    % Plot left image with edges
    nexttile;
    imshow(img_left_crop); hold on;
    title(sprintf('Left Image - Region %d (%.1f, %.1f) [%d edges]', ...
                  region_idx, x_start, y_start, length(edges_in_region)));
    
    for i = 1:length(edges_in_region)
        x = edges_in_region(i).left_x + 1;  
        y = edges_in_region(i).left_y + 1;
        theta = edges_in_region(i).left_ori;
        
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        plot([x - dx/2, x + dx/2], [y - dy/2, y + dy/2], 'Color', colors(i, :), 'LineWidth', 6);
    end
    hold off;
    
    % Plot right image with edges
    nexttile;
    imshow(img_right_crop); hold on;
    title(sprintf('Right Image - Region %d (%.1f, %.1f) [%d edges]', ...
                  region_idx, x_start, y_start, length(edges_in_region)));
    
    for i = 1:length(edges_in_region)
        x = edges_in_region(i).right_x + 1;  
        y = edges_in_region(i).right_y + 1;
        theta = edges_in_region(i).right_ori;
        
        dx = edge_length * cos(theta);
        dy = edge_length * sin(theta);
        
        plot([x - dx/2, x + dx/2], [y - dy/2, y + dy/2], 'Color', colors(i, :), 'LineWidth', 6);
    end
    hold off;
    
    % Save figure as PDF with exact dimensions
    pdf_path = sprintf('../outputs/stereo_matches_tp_region_%d.pdf', region_idx);
    
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0, 0, target_width/100, target_height/100]);
    set(fig, 'PaperSize', [target_width/100, target_height/100]);
    
    print(fig, pdf_path, '-dpdf', '-r300');
    fprintf('Saved: %s\n', pdf_path);
end