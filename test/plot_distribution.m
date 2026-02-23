function plot_distribution(filter_name, frame_idx, output_dir)

% PLOT_DISTRIBUTION Plot PDF and CDF for filter distribution data

% Plots separate distributions for veridical (GT) and non-veridical edges

% Saves PDF and CDF as separate image files.



if nargin < 2

frame_idx = 0;

end

if nargin < 3

output_dir = '../output_files';

end


% Construct filename

filename = fullfile(output_dir, sprintf('%s_frame_%d.txt', filter_name, frame_idx));


% Check if file exists

if ~exist(filename, 'file')

error('File not found: %s', filename);

end


% Read the data (skip header lines starting with #)

fid = fopen(filename, 'r');

data = [];

is_gt = [];

header_done = false;


while ~feof(fid)

line = fgetl(fid);

if ischar(line) && ~isempty(line)

if line(1) == '#'

continue;

elseif ~header_done && contains(line, 'filter_value')

header_done = true;

continue;

elseif header_done

parts = strsplit(line, '\t');

if length(parts) >= 2

value = str2double(parts{1});

gt_flag = str2double(parts{2});

if ~isnan(value) && ~isnan(gt_flag)

data = [data; value];

is_gt = [is_gt; gt_flag];

end

end

end

end

end

fclose(fid);


if isempty(data)

error('No valid data found in file: %s', filename);

end


% Transform NCC scores to dissimilarity (1 - ncc_score)

is_ncc_filter = contains(filter_name, 'ncc_score');

if is_ncc_filter || strcmp(filter_name, 'ncc_score')

data = 1 - data;

filter_display_name = 'NCC dissimilarity (1 - NCC)';

elseif strcmp(filter_name, 'location_error')

filter_display_name = 'disparity difference';

else

filter_display_name = strrep(filter_name, '_', ' ');

end


% Separate GT and non-GT data

data_gt = data(is_gt == 1);

data_non_gt = data(is_gt == 0);


fprintf('Loaded %d values from %s\n', length(data), filename);


% Check which filter to determine if zoomed inset plot is needed

is_epipolar = strcmp(filter_name, 'epipolar_distance');

is_location = strcmp(filter_name, 'location_error');


%% --- PLOT 1: PDF (With Zoomed Inset) ---

% Create figure for PDF with larger size

fig_pdf = figure('Position', [100, 100, 1200, 900]);

ax_main = axes('Parent', fig_pdf); % Explicitly create main axes

hold(ax_main, 'on');


% Define common bin edges

data_min = min([data_gt; data_non_gt]);

data_max = max([data_gt; data_non_gt]);

bin_edges = linspace(data_min, data_max, 51);


% Plot main histograms explicitly to ax_main

histogram(ax_main, data_non_gt, bin_edges, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.2, 0.2], ...

'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Non-veridical');

histogram(ax_main, data_gt, bin_edges, 'Normalization', 'pdf', 'FaceColor', [0.2, 0.6, 0.8], ...

'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Veridical');


xlabel(ax_main, filter_display_name, 'FontSize', 32, 'FontWeight', 'bold');

ylabel(ax_main, 'Probability Density', 'FontSize', 32, 'FontWeight', 'bold');

title(ax_main, sprintf('PDF - %s (Frame %d)', filter_display_name, frame_idx), 'FontSize', 36, 'FontWeight', 'bold');


% Legend and Text

leg = legend(ax_main, 'Location', 'northeast', 'FontSize', 28);

leg.Position(2) = leg.Position(2) - 0.15;

grid(ax_main, 'on');

set(ax_main, 'FontSize', 28, 'LineWidth', 2);

ax_main.XAxis.LineWidth = 2.5;

ax_main.YAxis.LineWidth = 2.5;


text_str = sprintf('Veridical: n=%d, μ=%.3f, σ=%.3f\nNon-veridical: n=%d, μ=%.3f, σ=%.3f', ...

length(data_gt), mean(data_gt), std(data_gt), ...

length(data_non_gt), mean(data_non_gt), std(data_non_gt));

text(ax_main, 0.98, 0.98, text_str, 'Units', 'normalized', ...

'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...

'FontSize', 24, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...

'Interpreter', 'none', 'LineWidth', 2);


hold(ax_main, 'off');


% --- INSET LOGIC ---

% Calculate position based on the main axes - MOVED DOWNWARD

main_pos = get(ax_main, 'Position'); % [left, bottom, width, height]


inset_width = main_pos(3) * 0.38;

inset_height = main_pos(4) * 0.35;

% Keep horizontal position, just move down vertically

inset_left = main_pos(1) + main_pos(3) * 0.32; % Same as before

inset_bottom = main_pos(2) + main_pos(4) * 0.15; % Moved down from 0.42

% Initialize variables for zoom logic

plot_inset = false;

zoom_min = 0; zoom_max = 0;

zoom_title = '';


if is_ncc_filter

plot_inset = true;

zoom_min = 0; zoom_max = 0.5;

zoom_title = 'Zoomed: 1-NCC [0, 0.5]';

elseif is_epipolar

plot_inset = true;

zoom_min = 0; zoom_max = 2;

zoom_title = 'Zoomed: EP 0-2 pixels';

elseif is_location

plot_inset = true;

zoom_min = 0; zoom_max = 20;

zoom_title = 'Zoomed: LP 0-20 pixels';

end


if plot_inset

% Create inset axes explicitly on the PDF figure

ax_inset = axes('Parent', fig_pdf, 'Position', [inset_left, inset_bottom, inset_width, inset_height]);

hold(ax_inset, 'on');


% Filter data

data_gt_zoom = data_gt(data_gt >= zoom_min & data_gt <= zoom_max);

data_non_gt_zoom = data_non_gt(data_non_gt >= zoom_min & data_non_gt <= zoom_max);

bin_edges_zoom = linspace(zoom_min, zoom_max, 31);


% Plot Histograms explicitly to ax_inset

histogram(ax_inset, data_non_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.2, 0.2], ...

'EdgeColor', 'none', 'FaceAlpha', 0.6);

histogram(ax_inset, data_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', 'FaceColor', [0.2, 0.6, 0.8], ...

'EdgeColor', 'none', 'FaceAlpha', 0.6);


% Styling

xlabel(ax_inset, filter_display_name, 'FontSize', 22, 'FontWeight', 'bold');

ylabel(ax_inset, 'PDF', 'FontSize', 22, 'FontWeight', 'bold');

title(ax_inset, zoom_title, 'FontSize', 24, 'FontWeight', 'bold');

grid(ax_inset, 'on');

xlim(ax_inset, [zoom_min, zoom_max]);

set(ax_inset, 'FontSize', 20, 'LineWidth', 2);

ax_inset.XAxis.LineWidth = 2;

ax_inset.YAxis.LineWidth = 2;

box(ax_inset, 'on');


% Add white background to make it stand out

ax_inset.Color = [1 1 1];

ax_inset.Box = 'on';

ax_inset.LineWidth = 3;


hold(ax_inset, 'off');




% --- FIXED ARROW LOGIC ---


% 1. Define the "target" in the main plot (the center of the zoomed region)

target_data_x = (zoom_min + zoom_max) / 2;


% 2. Convert Main Axes Data coordinates to Figure Normalized coordinates

% This ensures the arrow points exactly at the bin location on the X-axis

ax_main.Units = 'normalized'; % Ensure we are working in normalized units

xl = xlim(ax_main);


% Calculate where 'target_data_x' sits as a percentage of the main axis width

x_ratio = (target_data_x - xl(1)) / (xl(2) - xl(1));


% Map that ratio to the actual figure position of the axes

arrow_end_x = main_pos(1) + (x_ratio * main_pos(3));


% We want it to point to the "floor" of the main plot (the X-axis line)

arrow_end_y = main_pos(2);


% 3. Define the start point (Bottom-Left or Bottom-Center of the inset)

% Starting from the bottom of the inset creates a "pointer" effect

arrow_start_x = inset_left;

arrow_start_y = inset_bottom;



% Create the annotation

annotation(fig_pdf, 'arrow', [arrow_start_x, arrow_end_x], [arrow_start_y, arrow_end_y], ...

'LineWidth', 3.5, 'Color', [0.2, 0.2, 0.2], ...

'HeadStyle', 'cback1', 'HeadLength', 15, 'HeadWidth', 15); end

% Save PDF Figure

save_filename_pdf = fullfile(output_dir, sprintf('%s_frame_%d_pdf.png', filter_name, frame_idx));

saveas(fig_pdf, save_filename_pdf);

fprintf('Saved PDF figure to: %s\n', save_filename_pdf);

close(fig_pdf);


%% --- PLOT 2: CDF (Final Publication Ready - Refined Legend) ---

fig_cdf = figure('Position', [150, 150, 1200, 900]);

ax_cdf = axes('Parent', fig_cdf);

hold(ax_cdf, 'on');


% 1. CALCULATE ECDF

[f_non_gt, x_non_gt] = ecdf(data_non_gt);

[f_gt, x_gt] = ecdf(data_gt);


% 2. PLOT MAIN LINES

plot(ax_cdf, x_non_gt, f_non_gt*100, 'LineWidth', 7, 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'Non-veridical');

plot(ax_cdf, x_gt, f_gt*100, 'LineWidth', 7, 'Color', [0.2, 0.6, 0.8], 'DisplayName', 'Veridical');


% 3. CALCULATE PERCENTILES

percentiles = [50, 90, 95, 99];

p_gt = prctile(data_gt, percentiles);

p_non_gt = prctile(data_non_gt, percentiles);


% 4. STYLING & AXES

xlabel(ax_cdf, filter_display_name, 'FontSize', 36, 'FontWeight', 'bold');

ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 36, 'FontWeight', 'bold');

title(ax_cdf, sprintf('CDF - %s', filter_display_name), 'FontSize', 38, 'FontWeight', 'bold');


grid(ax_cdf, 'on');

set(ax_cdf, 'FontSize', 32, 'LineWidth', 3);


xl = xlim(ax_cdf);

xr = xl(2) - xl(1);