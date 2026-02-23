function visualize_kf_cf_projection(csv_path, kf_img_path, cf_img_path)
% visualize_kf_cf_projection
% Visualize: Keyframe edge (48660), GT-projected CF point, CF candidates,
% and veridical CF edges parsed from a debug CSV.
%
% Usage:
%   visualize_kf_cf_projection(csv_path, kf_img_path, cf_img_path)
%     - csv_path    : path to kf_cf_projection_edge_48660_<left|right>.csv
%     - kf_img_path : keyframe right image (frame 0, im1)
%     - cf_img_path : current frame right image (frame 1, im1)
%
% Example:
%   csv_path = fullfile('..','outputs','kf_cf_projection_edge_48660_right.csv');
%   kf_img   = fullfile('..','outputs','image_verification','prev_frame_1.png');
%   cf_img   = fullfile('..','outputs','image_verification','curr_frame_1.png');
%   visualize_kf_cf_projection(csv_path, kf_img, cf_img);

if nargin < 1 || isempty(csv_path)
    error('csv_path is required.');
end

txt = fileread(csv_path);

% Parse KF edge line: "kf_edge_index" then "48660:(x y ori)"
kf_block = extract_section_line(txt, 'kf_edge_index');
if isempty(kf_block)
    error('Missing kf_edge_index section in %s', csv_path);
end
[kf_xy, kf_ori] = parse_single_index_xy(kf_block);

% Parse GT projection line: "gt_projected_cf" then "(x y ori)"
proj_line = extract_section_line(txt, 'gt_projected_cf');
if isempty(proj_line)
    error('Missing gt_projected_cf section in %s', csv_path);
end
[proj_xy, proj_ori] = parse_xy_no_index(proj_line);

% Parse candidates list
cand_list = extract_section_list(txt, 'matching_cf_edges_indices');
[cand_pts, cand_oris]  = parse_index_coord_list(cand_list);

% Parse veridicals list
ver_list = extract_section_list(txt, 'veridical_cf_edges_indices');
[ver_pts, ver_oris]  = parse_index_coord_list(ver_list);

fig = figure('Name','KF/CF Right Images with Markers','Color','w');

% Left subplot: Keyframe right image with KF edge marked
ax1 = subplot(1,2,1);
if nargin >= 2 && ~isempty(kf_img_path) && exist(kf_img_path,'file')
    kf_img = imread(kf_img_path);
    imshow(kf_img, [], 'Parent', ax1);
else
    warning('KF image not provided or not found. Showing blank axes.');
    axes(ax1); cla(ax1); axis(ax1,'ij'); axis(ax1,'equal');
end
hold(ax1, 'on'); grid(ax1, 'on');
title(ax1, 'Keyframe Right (frame0, im1)');
% Draw KF edge as oriented line with center marker
draw_oriented_edge(ax1, kf_xy, kf_ori, 'r', 1.5, 5, 'KF edge (48660)');
legend(ax1, 'Location','best');

% Right subplot: Current frame right image with CF candidates, veridicals, and GT projection
ax2 = subplot(1,2,2);
if nargin >= 3 && ~isempty(cf_img_path) && exist(cf_img_path,'file')
    cf_img = imread(cf_img_path);
    imshow(cf_img, [], 'Parent', ax2);
else
    warning('CF image not provided or not found. Showing blank axes.');
    axes(ax2); cla(ax2); axis(ax2,'ij'); axis(ax2,'equal');
end
hold(ax2, 'on'); grid(ax2, 'on');
title(ax2, 'Current Right (frame1, im1)');
% Draw GT projection as oriented line with center marker
draw_oriented_edge(ax2, proj_xy, proj_ori, 'm', 1.5, 5, 'GT projection');
% Draw candidates
if ~isempty(cand_pts)
    for i = 1:size(cand_pts,1)
        draw_oriented_edge(ax2, cand_pts(i,:), cand_oris(i), 'b', 0.8, 4, '');
    end
    % Add dummy for legend
    plot(ax2, NaN, NaN, 'b-', 'LineWidth', 0.8, 'DisplayName', sprintf('CF candidates (%d)', size(cand_pts,1)));
end
% Draw veridicals
if ~isempty(ver_pts)
    for i = 1:size(ver_pts,1)
        draw_oriented_edge(ax2, ver_pts(i,:), ver_oris(i), 'g', 1.2, 4.5, '');
    end
    % Add dummy for legend
    plot(ax2, NaN, NaN, 'g-', 'LineWidth', 1.2, 'DisplayName', sprintf('CF veridicals (%d)', size(ver_pts,1)));
end
legend(ax2, 'Location','best');

end

% Helper to draw oriented edge as line segment with center marker
function draw_oriented_edge(ax, xy, ori, color, linewidth, len, label)
% xy: [x, y] center
% ori: orientation in radians
% len: line length in pixels
dx = len/2 * cos(ori);
dy = len/2 * sin(ori);
x1 = xy(1) - dx; y1 = xy(2) - dy;
x2 = xy(1) + dx; y2 = xy(2) + dy;
if isempty(label)
    plot(ax, [x1 x2], [y1 y2], '-', 'Color', color, 'LineWidth', linewidth, 'HandleVisibility','off');
    plot(ax, xy(1), xy(2), 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerSize', 3, 'HandleVisibility','off');
else
    plot(ax, [x1 x2], [y1 y2], '-', 'Color', color, 'LineWidth', linewidth, 'DisplayName', label);
    plot(ax, xy(1), xy(2), 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerSize', 4, 'HandleVisibility','off');
end
end

% ---- Helpers ----
function line_text = extract_section_line(txt, section_label)
lines = regexp(txt, '\r?\n', 'split');
idx = find(strcmp(strtrim(lines), strtrim(section_label)), 1, 'first');
if isempty(idx) || idx+1 > numel(lines)
    line_text = '';
    return;
end
line_text = strtrim(lines{idx+1});
end

function list_text = extract_section_list(txt, section_label)
lines = regexp(txt, '\r?\n', 'split');
idx = find(strcmp(strtrim(lines), strtrim(section_label)), 1, 'first');
if isempty(idx) || idx+1 > numel(lines)
    list_text = '';
    return;
end
list_text = strtrim(lines{idx+1});
end

function [xy, ori] = parse_single_index_xy(line_text)
% Expect: "48660:(x y ori)"
tokens = regexp(line_text, '^[0-9]+:\(([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\)$', 'tokens');
if isempty(tokens)
    error('Failed to parse KF index line: %s', line_text);
end
xy = [str2double(tokens{1}{1}), str2double(tokens{1}{2})];
ori = str2double(tokens{1}{3});
end

function [xy, ori] = parse_xy_no_index(line_text)
% Expect: "(x y ori)"
tokens = regexp(line_text, '^\(([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\)$', 'tokens');
if isempty(tokens)
    error('Failed to parse projection line: %s', line_text);
end
xy = [str2double(tokens{1}{1}), str2double(tokens{1}{2})];
ori = str2double(tokens{1}{3});
end

function [pts, oris] = parse_index_coord_list(list_text)
% Parses "index:(x y ori);index:(x y ori);..." into Nx2 [x y] and Nx1 ori
pts = []; oris = [];
if isempty(list_text)
    return;
end
items = regexp(list_text, ';', 'split');
xy = nan(numel(items), 2);
ori_vec = nan(numel(items), 1);
n = 0;
for k = 1:numel(items)
    item = strtrim(items{k});
    if item == ""; continue; end
    tokens = regexp(item, '^[0-9]+:\(([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\)$', 'tokens');
    if isempty(tokens); continue; end
    x = str2double(tokens{1}{1}); y = str2double(tokens{1}{2}); ori_val = str2double(tokens{1}{3});
    if ~isnan(x) && ~isnan(y) && ~isnan(ori_val)
        n = n + 1; xy(n,:) = [x,y]; ori_vec(n) = ori_val;
    end
end
pts = xy(1:n, :);
oris = ori_vec(1:n);
end
