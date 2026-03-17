%% visualize_3d_edges.m
%  Reads edge_numbers.txt and visualizes 3D edge points colored by RGB
%  with tangent line segments showing the edge orientation.
%  Also loads stereo images and plots 2D projected tangents.

clear; close all; clc;

%% ---- Dataset & Image Paths ---------------------------------------------
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";

% For frame 0 (previous frame)
stereo_pair_0 = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";

img_path_0_left = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im0.png";
img_0_left = imread(char(img_path_0_left));

img_path_0_right = source_dataset_folder + dataset_sequence_path + stereo_pair_0 + "/im1.png";
img_0_right = imread(char(img_path_0_right));


%% ---- Parameters --------------------------------------------------------
filename        = '../outputs/edge_numbers.txt';
tangent_len     = 0.04;       % half-length of 3D tangent line segment
tangent_len_2d  = 5;         % half-length of 2D tangent line segment (in pixels)
marker_size     = 40;         % scatter point size
marker_size_2d  = 20;         % scatter point size for 2D images
tangent_width   = 1.5;        % line width of tangent segments
show_tangents   = true;       % toggle tangent lines on/off
point_color     = [0.2, 0.4, 0.9];  % blue for points
tangent_color   = [0.9, 0.2, 0.2];  % red for tangents

%% ---- Parse the file ----------------------------------------------------
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open file: %s', filename);
end

positions          = [];   % Nx3  (x, y, z)
orientations       = [];   % Nx3  (ox, oy, oz)
proj_tangent_left  = [];   % Nx2  (tx, ty) 2D left tangent
proj_tangent_right = [];   % Nx2  (tx, ty) 2D right tangent
colors             = [];   % Nx3  (R, G, B)
regions            = [];   % Nx1  region id
pixels_left        = [];   % Nx2  (px, py) Left image pixel
pixels_right       = [];   % Nx2  (px, py) Right image pixel

% Regex pattern for the new format including Right Pixel
pattern = 'Region:\s*(\d+).*Pixel:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*Right Pixel:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*GT Location\s*:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*orientation:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*projected tangent left:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*projected tangent right:\s*\(\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*\).*RGB:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)';

while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line)
        continue;
    end

    tokens = regexp(line, pattern, 'tokens');

    if ~isempty(tokens)
        vals = str2double(tokens{1});
        regions            = [regions;            vals(1)];
        
        % Convert 0-based C++ pixel coordinates to 1-based MATLAB coordinates
        pixels_left        = [pixels_left;        vals(2) + 1, vals(3) + 1];
        pixels_right       = [pixels_right;       vals(4) + 1, vals(5) + 1];
        
        positions          = [positions;          vals(6), vals(7), vals(8)];
        orientations       = [orientations;       vals(9), vals(10), vals(11)];
        
        % Extract X, Y for left and right 2D tangents, ignore Z
        proj_tangent_left  = [proj_tangent_left;  vals(12), vals(13)]; 
        proj_tangent_right = [proj_tangent_right; vals(15), vals(16)]; 
        
        colors             = [colors;             vals(18), vals(19), vals(20)];
    end
end
fclose(fid);

N = size(positions, 1);
fprintf('Loaded %d edge points.\n', N);

%% ---- Normalize 3D and 2D tangent vectors to unit length ----------------
% 3D
norms3D = sqrt(sum(orientations.^2, 2)); norms3D(norms3D == 0) = 1;
tangents3D = orientations ./ norms3D;

% 2D Left
norms2D_L = sqrt(sum(proj_tangent_left.^2, 2)); norms2D_L(norms2D_L == 0) = 1;
t_left_2D = proj_tangent_left ./ norms2D_L;

% 2D Right
norms2D_R = sqrt(sum(proj_tangent_right.^2, 2)); norms2D_R(norms2D_R == 0) = 1;
t_right_2D = proj_tangent_right ./ norms2D_R;

%% ---- Figure 1: 3-D scatter with tangent lines (colored by RGB) --------
fig1 = figure('Name', '3D Edge Points with Tangents', 'NumberTitle', 'off', ...
              'Color', 'w', 'Position', [100 100 800 600]);
scatter3(positions(:,1), positions(:,2), positions(:,3), ...
         marker_size, point_color, 'filled');
hold on;
if show_tangents
    for i = 1:N
        p  = positions(i,:);
        d  = tangents3D(i,:);
        p1 = p - tangent_len * d;
        p2 = p + tangent_len * d;
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
              '-', 'Color', tangent_color, 'LineWidth', tangent_width);
    end
end
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3-D Edge Points with Tangent Orientations');
axis equal; grid on; rotate3d on;

%% ---- Figure 2: Left Image with Projected Tangents ---------------------
fig2 = figure('Name', 'Left Image 2D Tangents', 'NumberTitle', 'off', ...
              'Color', 'w', 'Position', [150 150 800 600]);
imshow(img_0_left); hold on;
title('Left Image: Replotted Left Tangents');

scatter(pixels_left(:,1), pixels_left(:,2), marker_size_2d, point_color, 'filled');

if show_tangents
    for i = 1:N
        px = pixels_left(i,1);
        py = pixels_left(i,2);
        dx = t_left_2D(i,1);
        dy = t_left_2D(i,2);
        
        plot([px - tangent_len_2d * dx, px + tangent_len_2d * dx], ...
             [py - tangent_len_2d * dy, py + tangent_len_2d * dy], ...
             '-', 'Color', tangent_color, 'LineWidth', tangent_width);
    end
end

%% ---- Figure 3: Right Image with Projected Tangents --------------------
fig3 = figure('Name', 'Right Image 2D Tangents', 'NumberTitle', 'off', ...
              'Color', 'w', 'Position', [200 200 800 600]);
imshow(img_0_right); hold on;
title('Right Image: Replotted Right Tangents');

scatter(pixels_right(:,1), pixels_right(:,2), marker_size_2d, point_color, 'filled');

if show_tangents
    for i = 1:N
        px = pixels_right(i,1);
        py = pixels_right(i,2);
        dx = t_right_2D(i,1);
        dy = t_right_2D(i,2);
        
        plot([px - tangent_len_2d * dx, px + tangent_len_2d * dx], ...
             [py - tangent_len_2d * dy, py + tangent_len_2d * dy], ...
             '-', 'Color', tangent_color, 'LineWidth', tangent_width);
    end
end

fprintf('Finished plotting 3D and 2D projected tangents.\n');