function visualize_veridical_chain(csv_path, kf_left_img, kf_right_img, cf_left_img, cf_right_img)
% visualize_veridical_chain
% Visualizes the complete correspondence chain with interactive edge selection
% 
% Layout (2 rows: 2 on top, 3 on bottom):
%   Top row:
%     A- (KF left edge) - Red
%     B- (KF right stereo mates) - Cyan
%   Bottom row:
%     A+ (CF left veridical edges) - Green, with A- GT projection (orange star)
%     B+ via A+ path (CF right from A+) - Magenta
%     B+ via B- path (CF right from B-) - Blue, with B- GT projections (orange stars)
%
% Usage:
%   visualize_veridical_chain(csv_path, kf_left_img, kf_right_img, cf_left_img, cf_right_img)
%     - csv_path      : path to veridical_debug_left.csv
%     - kf_left_img   : keyframe left image (frame 0, im0)
%     - kf_right_img  : keyframe right image (frame 0, im1)
%     - cf_left_img   : current frame left image (frame 1, im0)
%     - cf_right_img  : current frame right image (frame 1, im1)
%
% Click on any edge to see its index, location, and orientation

if nargin < 1 || isempty(csv_path)
    error('csv_path is required.');
end

txt = fileread(csv_path);

% Parse A- (KF left edge)
a_minus_section = extract_section_until_next(txt, '(A-)KF Edge Index');
[a_minus_xy, a_minus_ori, a_minus_idx] = parse_single_edge(a_minus_section);

% Parse A- to A+ GT projection
a_minus_projected = [];
lines = strsplit(txt, '\n');
for i = 1:length(lines)
    line = strtrim(lines{i});
    if contains(line, 'left_gt_projected_location_on_cf_x')
        % Next line has the projected point data
        if i < length(lines)
            next_line = strtrim(lines{i+1});
            parts = strsplit(next_line, ',');
            if length(parts) >= 3
                proj_x = str2double(parts{1});
                proj_y = str2double(parts{2});
                proj_ori = str2double(parts{3});
                if ~isnan(proj_x) && ~isnan(proj_y) && ~isnan(proj_ori)
                    a_minus_projected = [proj_x, proj_y, proj_ori];
                end
            end
        end
        break;
    end
end

% Parse B- (KF right stereo mates)
b_minus_sections = extract_multiple_sections(txt, '(B-)GT Veridical Stereo KF Edges:');
b_minus_edges = [];
for i = 1:length(b_minus_sections)
    [xy, ori, idx] = parse_edge_list(b_minus_sections{i});
    if ~isempty(xy)
        b_minus_edges = [b_minus_edges; xy, ori, idx];
    end
end

% Parse A+ (CF left veridical edges)
a_plus_sections = extract_multiple_sections(txt, '(A+)GT Veridical CF Edges:');
a_plus_edges = [];
for i = 1:length(a_plus_sections)
    [xy, ori, idx] = parse_edge_list(a_plus_sections{i});
    if ~isempty(xy)
        a_plus_edges = [a_plus_edges; xy, ori, idx];
    end
end

% Parse B+ from both paths
b_plus_from_b = []; % B+ via B- (from "(B+)GT Veridical Stereo CF Edges:")
b_plus_from_a = []; % B+ via A+ (from "(B+) 2nd GT Veridical Stereo CF Edges:")
b_minus_projected = []; % Projected points from B- to CF (GT projection)

% Extract B+ sections carefully
lines = strsplit(txt, '\n');
in_b_plus = false;
current_type = '';
skip_next_data = false;

for i = 1:length(lines)
    line = strtrim(lines{i});
    
    % Check for projected point from B- (only right projections)
    if contains(line, 'right_gt_projected_location_on_cf_x')
        % Next line has the projected point data
        if i < length(lines)
            next_line = strtrim(lines{i+1});
            parts = strsplit(next_line, ',');
            if length(parts) >= 3
                proj_x = str2double(parts{1});
                proj_y = str2double(parts{2});
                proj_ori = str2double(parts{3});
                if ~isnan(proj_x) && ~isnan(proj_y) && ~isnan(proj_ori)
                    b_minus_projected = [b_minus_projected; proj_x, proj_y, proj_ori];
                end
            end
        end
        continue;
    end
    
    % Check for section headers
    if contains(line, '(B+)GT Veridical Stereo CF Edges:') && ~contains(line, '2nd')
        in_b_plus = true;
        current_type = 'from_b';
        skip_next_data = false;
        continue;
    elseif contains(line, '(B+) 2nd GT Veridical Stereo CF Edges:')
        in_b_plus = true;
        current_type = 'from_a';
        skip_next_data = false;
        continue;
    elseif contains(line, 'Index,Location')
        % Skip header line but allow next data lines
        continue;
    elseif contains(line, 'DEBUG:')
        % Skip debug lines
        continue;
    elseif contains(line, '(A+)') || contains(line, '(A-)') || contains(line, '(B-)')
        in_b_plus = false;
        current_type = '';
        skip_next_data = false;
        continue;
    end
    
    if in_b_plus && ~isempty(line)
        parts = strsplit(line, ',');
        if length(parts) >= 4
            idx = str2double(parts{1});
            x = str2double(parts{2});
            y = str2double(parts{3});
            ori = str2double(parts{4});
            if ~isnan(idx) && ~isnan(x) && ~isnan(y) && ~isnan(ori)
                if strcmp(current_type, 'from_b')
                    b_plus_from_b = [b_plus_from_b; x, y, ori, idx];
                elseif strcmp(current_type, 'from_a')
                    b_plus_from_a = [b_plus_from_a; x, y, ori, idx];
                end
            end
        end
    end
end

% Find overlapped B+ edges (present in both paths)
% First, get unique edges from each path
b_plus_from_a_unique = [];
b_plus_from_b_unique = [];

if ~isempty(b_plus_from_a)
    [~, unique_idx_a] = unique(b_plus_from_a(:,4), 'stable');
    b_plus_from_a_unique = b_plus_from_a(unique_idx_a, :);
end

if ~isempty(b_plus_from_b)
    [~, unique_idx_b] = unique(b_plus_from_b(:,4), 'stable');
    b_plus_from_b_unique = b_plus_from_b(unique_idx_b, :);
end

% Now find overlaps based on unique indices
b_plus_indices_from_a = [];
b_plus_indices_from_b = [];
if ~isempty(b_plus_from_a_unique)
    b_plus_indices_from_a = b_plus_from_a_unique(:,4);
end
if ~isempty(b_plus_from_b_unique)
    b_plus_indices_from_b = b_plus_from_b_unique(:,4);
end
overlapped_indices = intersect(b_plus_indices_from_a, b_plus_indices_from_b);
num_overlapped = length(overlapped_indices);

% Calculate total unique B+ edges across both paths
total_unique_b_plus = length(union(b_plus_indices_from_a, b_plus_indices_from_b));

% Separate unique edges into overlapped and path-specific
b_plus_from_a_only = [];
b_plus_from_b_only = [];
b_plus_overlapped = [];

if ~isempty(b_plus_from_a_unique)
    for i = 1:size(b_plus_from_a_unique,1)
        if ismember(b_plus_from_a_unique(i,4), overlapped_indices)
            b_plus_overlapped = [b_plus_overlapped; b_plus_from_a_unique(i,:)];
        else
            b_plus_from_a_only = [b_plus_from_a_only; b_plus_from_a_unique(i,:)];
        end
    end
end

if ~isempty(b_plus_from_b_unique)
    for i = 1:size(b_plus_from_b_unique,1)
        if ~ismember(b_plus_from_b_unique(i,4), overlapped_indices)
            b_plus_from_b_only = [b_plus_from_b_only; b_plus_from_b_unique(i,:)];
        end
    end
end

% Create figure with 2 rows: 2 on top, 3 on bottom
fig = figure('Name','Veridical Correspondence Chain - Click edges for details','Color','w','Position',[100 100 1800 1200]);

% A- (top left): Keyframe left image
ax1 = subplot(2,3,1);
if nargin >= 2 && ~isempty(kf_left_img) && exist(kf_left_img,'file')
    img = imread(kf_left_img);
    imshow(img, [], 'Parent', ax1);
else
    axes(ax1); axis(ax1,'ij'); axis(ax1,'equal');
end
hold(ax1, 'on'); grid(ax1, 'on');
title(ax1, sprintf('A- Keyframe Left (Edge %d)', a_minus_idx), 'FontSize', 14, 'FontWeight', 'bold');
% Draw edge with legend
draw_interactive_edge_with_legend(ax1, a_minus_xy, a_minus_ori, a_minus_idx, 'r', 2.5, 3, sprintf('A- Edge %d', a_minus_idx));
legend(ax1, 'Location','northeast', 'FontSize', 10);

% B- (top right): Keyframe right image - aligned with B+ via B- path below
ax2 = subplot(2,3,3);
if nargin >= 3 && ~isempty(kf_right_img) && exist(kf_right_img,'file')
    img = imread(kf_right_img);
    imshow(img, [], 'Parent', ax2);
else
    axes(ax2); axis(ax2,'ij'); axis(ax2,'equal');
end
hold(ax2, 'on'); grid(ax2, 'on');
title(ax2, sprintf('B- Keyframe Right (%d stereo mates)', size(b_minus_edges,1)), 'FontSize', 14, 'FontWeight', 'bold');
if ~isempty(b_minus_edges)
    for i = 1:size(b_minus_edges,1)
        draw_interactive_edge(ax2, b_minus_edges(i,1:2), b_minus_edges(i,3), b_minus_edges(i,4), 'c', 1.8, 3);
    end
    % Add single legend entry
    plot(ax2, NaN, NaN, 'c-', 'LineWidth', 1.8, 'DisplayName', sprintf('B- Stereo mates (%d)', size(b_minus_edges,1)));
end
legend(ax2, 'Location','northeast', 'FontSize', 10);

% A+ (bottom left): Current frame left image with GT projection
ax3 = subplot(2,3,4);
if nargin >= 4 && ~isempty(cf_left_img) && exist(cf_left_img,'file')
    img = imread(cf_left_img);
    imshow(img, [], 'Parent', ax3);
else
    axes(ax3); axis(ax3,'ij'); axis(ax3,'equal');
end
hold(ax3, 'on'); grid(ax3, 'on');
title(ax3, sprintf('A+ Current Left (%d veridical edges)', size(a_plus_edges,1)), 'FontSize', 14, 'FontWeight', 'bold');

% Draw A- to A+ GT projection (orange star)
if ~isempty(a_minus_projected)
    plot(ax3, a_minus_projected(1), a_minus_projected(2), 'p', ...
         'MarkerFaceColor', [1 0.6 0], 'MarkerEdgeColor', 'k', 'MarkerSize', 12, ...
         'DisplayName', 'A- GT projection');
    % Add orientation line
    ori = a_minus_projected(3);
    len = 4;
    dx = len * cos(ori);
    dy = len * sin(ori);
    plot(ax3, [a_minus_projected(1)-dx, a_minus_projected(1)+dx], ...
              [a_minus_projected(2)-dy, a_minus_projected(2)+dy], ...
              '-', 'Color', [1 0.6 0], 'LineWidth', 2, 'HandleVisibility', 'off');
end

if ~isempty(a_plus_edges)
    for i = 1:size(a_plus_edges,1)
        draw_interactive_edge(ax3, a_plus_edges(i,1:2), a_plus_edges(i,3), a_plus_edges(i,4), 'g', 1.8, 3);
    end
    % Add single legend entry
    plot(ax3, NaN, NaN, 'g-', 'LineWidth', 1.8, 'DisplayName', sprintf('A+ Veridical (%d)', size(a_plus_edges,1)));
end
legend(ax3, 'Location','northeast', 'FontSize', 10);

% B+ via A+ (bottom center): Current frame right image - A+ path
ax4 = subplot(2,3,5);
if nargin >= 5 && ~isempty(cf_right_img) && exist(cf_right_img,'file')
    img = imread(cf_right_img);
    imshow(img, [], 'Parent', ax4);
else
    axes(ax4); axis(ax4,'ij'); axis(ax4,'equal');
end
hold(ax4, 'on'); grid(ax4, 'on');
title(ax4, sprintf('B+ via A+ path (%d edges, %d overlap)', size(b_plus_from_a_unique,1), num_overlapped), 'FontSize', 14, 'FontWeight', 'bold');

% Draw B+ from A+ only in magenta
if ~isempty(b_plus_from_a_only)
    for i = 1:size(b_plus_from_a_only,1)
        draw_interactive_edge(ax4, b_plus_from_a_only(i,1:2), b_plus_from_a_only(i,3), b_plus_from_a_only(i,4), 'm', 2.0, 3);
    end
    plot(ax4, NaN, NaN, 'm-', 'LineWidth', 2.0, 'DisplayName', sprintf('B+ via A+ only (%d)', size(b_plus_from_a_only,1)));
end

% Draw overlapped B+ in yellow/gold
if ~isempty(b_plus_overlapped)
    for i = 1:size(b_plus_overlapped,1)
        draw_interactive_edge(ax4, b_plus_overlapped(i,1:2), b_plus_overlapped(i,3), b_plus_overlapped(i,4), [1 0.84 0], 2.5, 3);
    end
    plot(ax4, NaN, NaN, '-', 'Color', [1 0.84 0], 'LineWidth', 2.5, 'DisplayName', sprintf('B+ both paths (%d)', size(b_plus_overlapped,1)));
end

legend(ax4, 'Location','northeast', 'FontSize', 10);

% B+ via B- (bottom right): Current frame right image - B- path
ax5 = subplot(2,3,6);
if nargin >= 5 && ~isempty(cf_right_img) && exist(cf_right_img,'file')
    img = imread(cf_right_img);
    imshow(img, [], 'Parent', ax5);
else
    axes(ax5); axis(ax5,'ij'); axis(ax5,'equal');
end
hold(ax5, 'on'); grid(ax5, 'on');
title(ax5, sprintf('B+ via B- path (%d edges, %d overlap)', size(b_plus_from_b_unique,1), num_overlapped), 'FontSize', 14, 'FontWeight', 'bold');

% Draw projected points from B- (yellow/orange stars) - GT projection
if ~isempty(b_minus_projected)
    for i = 1:size(b_minus_projected,1)
        % Draw as star marker with orientation line
        plot(ax5, b_minus_projected(i,1), b_minus_projected(i,2), 'p', ...
             'MarkerFaceColor', [1 0.6 0], 'MarkerEdgeColor', 'k', 'MarkerSize', 10, ...
             'HandleVisibility', 'off');
        % Add orientation line (no legend)
        ori = b_minus_projected(i,3);
        len = 4;
        dx = len * cos(ori);
        dy = len * sin(ori);
        plot(ax5, [b_minus_projected(i,1)-dx, b_minus_projected(i,1)+dx], ...
                  [b_minus_projected(i,2)-dy, b_minus_projected(i,2)+dy], ...
                  '-', 'Color', [1 0.6 0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    % Add single legend entry for all B- projected points
    plot(ax5, NaN, NaN, 'p', 'MarkerFaceColor', [1 0.6 0], 'MarkerEdgeColor', 'k', ...
         'MarkerSize', 10, 'DisplayName', sprintf('B- GT projection (%d)', size(b_minus_projected,1)));
end

% Draw B+ from B- only in blue
if ~isempty(b_plus_from_b_only)
    for i = 1:size(b_plus_from_b_only,1)
        draw_interactive_edge(ax5, b_plus_from_b_only(i,1:2), b_plus_from_b_only(i,3), b_plus_from_b_only(i,4), 'b', 1.5, 3);
    end
    plot(ax5, NaN, NaN, 'b-', 'LineWidth', 1.5, 'DisplayName', sprintf('B+ via B- only (%d)', size(b_plus_from_b_only,1)));
end

% Draw overlapped B+ in yellow/gold
if ~isempty(b_plus_overlapped)
    for i = 1:size(b_plus_overlapped,1)
        draw_interactive_edge(ax5, b_plus_overlapped(i,1:2), b_plus_overlapped(i,3), b_plus_overlapped(i,4), [1 0.84 0], 2.5, 3);
    end
    plot(ax5, NaN, NaN, '-', 'Color', [1 0.84 0], 'LineWidth', 2.5, 'DisplayName', sprintf('B+ both paths (%d)', size(b_plus_overlapped,1)));
end

legend(ax5, 'Location','northeast', 'FontSize', 10);

% Enable data cursor mode for all subplots
dcm = datacursormode(fig);
set(dcm, 'UpdateFcn', @edge_cursor_callback);
set(dcm, 'Enable', 'on');

fprintf('\nVisualization complete! Click on any edge to see details.\n');

end

%% Helper functions

function [xy, ori, idx] = parse_single_edge(section)
% Parse a single edge from a section like "49863,451.35,478.90,1.98"
lines = strsplit(section, '\n');
xy = [];
ori = [];
idx = [];
for i = 1:length(lines)
    line = strtrim(lines{i});
    if ~contains(line, 'Index') && ~isempty(line)
        parts = strsplit(line, ',');
        if length(parts) >= 4
            idx = str2double(parts{1});
            x = str2double(parts{2});
            y = str2double(parts{3});
            ori = str2double(parts{4});
            if ~isnan(idx) && ~isnan(x) && ~isnan(y)
                xy = [x, y];
                return;
            end
        end
    end
end
end

function [xys, oris, idxs] = parse_edge_list(section)
% Parse multiple edges from a section
lines = strsplit(section, '\n');
xys = [];
oris = [];
idxs = [];
for i = 1:length(lines)
    line = strtrim(lines{i});
    if ~contains(line, 'Index') && ~contains(line, 'GT Veridical') && ~isempty(line) && ~contains(line, 'DEBUG:')
        parts = strsplit(line, ',');
        if length(parts) >= 4
            idx = str2double(parts{1});
            x = str2double(parts{2});
            y = str2double(parts{3});
            ori = str2double(parts{4});
            if ~isnan(idx) && ~isnan(x) && ~isnan(y)
                xys = [xys; x, y];
                oris = [oris; ori];
                idxs = [idxs; idx];
            end
        end
    end
end
end

function section = extract_section_until_next(txt, header)
% Extract section from header until next major section
idx = strfind(txt, header);
if isempty(idx)
    section = '';
    return;
end
start_pos = idx(1) + length(header);
% Find next section marker
next_markers = {'\n(A-)', '\n(B-)', '\n(A+)', '\n(B+)'};
end_pos = length(txt);
for i = 1:length(next_markers)
    next_idx = strfind(txt(start_pos:end), next_markers{i});
    if ~isempty(next_idx)
        end_pos = min(end_pos, start_pos + next_idx(1) - 1);
    end
end
section = txt(start_pos:end_pos);
end

function sections = extract_multiple_sections(txt, header)
% Extract all sections with given header
sections = {};
lines = strsplit(txt, '\n');
in_section = false;
current_section = {};
for i = 1:length(lines)
    line = lines{i};
    if contains(line, header)
        if in_section && ~isempty(current_section)
            sections{end+1} = strjoin(current_section, '\n');
        end
        in_section = true;
        current_section = {};
    elseif in_section
        if contains(line, '(A-)') || contains(line, '(B-)') || contains(line, '(A+)') || contains(line, '(B+)')
            sections{end+1} = strjoin(current_section, '\n');
            in_section = false;
            current_section = {};
        else
            current_section{end+1} = line;
        end
    end
end
if in_section && ~isempty(current_section)
    sections{end+1} = strjoin(current_section, '\n');
end
end

function draw_interactive_edge(ax, xy, ori, idx, color, linewidth, len)
% Draw oriented edge with center marker, clickable with metadata
% xy: [x, y] center
% ori: orientation in radians
% idx: edge index
% len: line length in pixels
dx = len * cos(ori);
dy = len * sin(ori);
x1 = xy(1) - dx;
y1 = xy(2) - dy;
x2 = xy(1) + dx;
y2 = xy(2) + dy;

% Draw line
h_line = plot(ax, [x1 x2], [y1 y2], '-', 'Color', color, 'LineWidth', linewidth, 'HandleVisibility', 'off');
% Store metadata in UserData
h_line.UserData = struct('idx', idx, 'x', xy(1), 'y', xy(2), 'ori', ori, 'ori_deg', rad2deg(ori));

% Draw center marker
h_marker = plot(ax, xy(1), xy(2), 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'HandleVisibility', 'off');
h_marker.UserData = struct('idx', idx, 'x', xy(1), 'y', xy(2), 'ori', ori, 'ori_deg', rad2deg(ori));
end

function draw_interactive_edge_with_legend(ax, xy, ori, idx, color, linewidth, len, legend_label)
% Draw oriented edge with center marker AND add to legend
% xy: [x, y] center
% ori: orientation in radians
% idx: edge index
% len: line length in pixels
% legend_label: label for legend
dx = len * cos(ori);
dy = len * sin(ori);
x1 = xy(1) - dx;
y1 = xy(2) - dy;
x2 = xy(1) + dx;
y2 = xy(2) + dy;

% Draw line with legend entry
h_line = plot(ax, [x1 x2], [y1 y2], '-', 'Color', color, 'LineWidth', linewidth, 'DisplayName', legend_label);
% Store metadata in UserData
h_line.UserData = struct('idx', idx, 'x', xy(1), 'y', xy(2), 'ori', ori, 'ori_deg', rad2deg(ori));

% Draw center marker (no legend)
h_marker = plot(ax, xy(1), xy(2), 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'HandleVisibility', 'off');
h_marker.UserData = struct('idx', idx, 'x', xy(1), 'y', xy(2), 'ori', ori, 'ori_deg', rad2deg(ori));
end

function txt = edge_cursor_callback(~, event_obj)
% Callback for data cursor to display edge information
target = get(event_obj, 'Target');
if isfield(target.UserData, 'idx')
    data = target.UserData;
    txt = {sprintf('Edge Index: %d', data.idx), ...
           sprintf('Location: (%.2f, %.2f)', data.x, data.y), ...
           sprintf('Orientation: %.3f rad (%.1f°)', data.ori, data.ori_deg)};
else
    pos = get(event_obj, 'Position');
    txt = {sprintf('X: %.2f', pos(1)), sprintf('Y: %.2f', pos(2))};
end
end
