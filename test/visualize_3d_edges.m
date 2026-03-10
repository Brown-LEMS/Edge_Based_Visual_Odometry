%% visualize_3d_edges.m
%  Reads edge_numbers.txt and visualizes 3D edge points colored by RGB
%  with tangent line segments showing the edge orientation.

clear; close all; clc;

%% ---- Parameters --------------------------------------------------------
filename        = '../outputs/edge_numbers.txt';
tangent_len     = 0.04;      % half-length of tangent line segment
marker_size     = 40;         % scatter point size
tangent_width   = 1.5;        % line width of tangent segments
show_tangents   = true;       % toggle tangent lines on/off
tangent_color_mode = 'rgb';   % 'rgb' = same color as point, 'gray' = uniform gray

%% ---- Parse the file ----------------------------------------------------
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open file: %s', filename);
end

positions    = [];   % Nx3  (x, y, z)
orientations = [];   % Nx3  (ox, oy, oz) — 3D unit tangent vectors
colors       = [];   % Nx3  (R, G, B) in [0,255]

while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line)
        continue;
    end

    tokens = regexp(line, ...
        'GT Location\s*:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*orientation:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*RGB:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', ...
        'tokens');

    if isempty(tokens)
        continue;
    end
    vals = str2double(tokens{1});
    positions    = [positions;    vals(1), vals(2), vals(3)];
    orientations = [orientations; vals(4), vals(5), vals(6)];
    colors       = [colors;       vals(7), vals(8), vals(9)];
end
fclose(fid);

N = size(positions, 1);
fprintf('Loaded %d edge points.\n', N);

%% ---- Normalize tangent vectors to unit length --------------------------
norms = sqrt(sum(orientations.^2, 2));
norms(norms == 0) = 1;  % avoid division by zero
tangents = orientations ./ norms;

%% ---- Normalize RGB to [0,1] for MATLAB --------------------------------
rgb01 = colors / 255.0;

%% ---- Figure 1: 3-D scatter with tangent lines -------------------------
fig1 = figure('Name', '3D Edge Points with Tangents', 'NumberTitle', 'off', ...
              'Color', 'w', 'Position', [100 100 1100 750]);

% Draw points
scatter3(positions(:,1), positions(:,2), positions(:,3), ...
         marker_size, rgb01, 'filled');
hold on;

% Draw tangent line segments centered on each point: P - t*d  to  P + t*d
if show_tangents
    for i = 1:N
        p  = positions(i,:);
        d  = tangents(i,:);
        p1 = p - tangent_len * d;   % tail
        p2 = p + tangent_len * d;   % head

        if strcmp(tangent_color_mode, 'rgb')
            c = rgb01(i,:);
        else
            c = [0.4, 0.4, 0.4];
        end

        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
              '-', 'Color', c, 'LineWidth', tangent_width);
    end
end

xlabel('X'); ylabel('Y'); zlabel('Z');
title('3-D Edge Points with Tangent Orientations (colored by RGB)');
axis equal; grid on;
set(gca, 'FontSize', 12, 'Projection', 'perspective');
rotate3d on;

%% ---- Figure 2: top-down (X-Z) view with tangent lines -----------------
fig2 = figure('Name', 'Top-Down View (X-Z) with Tangents', 'NumberTitle', 'off', ...
              'Color', 'w', 'Position', [150 150 900 650]);

scatter(positions(:,1), positions(:,3), marker_size, rgb01, 'filled');
hold on;

if show_tangents
    for i = 1:N
        p  = positions(i,:);
        d  = tangents(i,:);
        p1 = p - tangent_len * d;
        p2 = p + tangent_len * d;

        if strcmp(tangent_color_mode, 'rgb')
            c = rgb01(i,:);
        else
            c = [0.4, 0.4, 0.4];
        end

        plot([p1(1) p2(1)], [p1(3) p2(3)], ...
             '-', 'Color', c, 'LineWidth', tangent_width);
    end
end

xlabel('X'); ylabel('Z');
title('Top-Down View (X vs Z) with Tangent Orientations');
axis equal; grid on;
set(gca, 'FontSize', 12);

fprintf('Done.\n');
