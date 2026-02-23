% PLOT_ALL_DISTRIBUTIONS - Plot all filter distributions for a given frame
%
% This script plots PDF and CDF for all three filters:
% - location_error
% - ncc_score
% - sift_distance

close all;
clear;
clc;

% Configuration
frame_idx = 0;  % Change this to plot different frames
output_dir = '../output_files';  % Change this to the appropriate output directory

% List of filters to plot
filters = {'epipolar_distance'};

% List of stages for ambiguity plots (match output file names exactly)
ambiguity_stages = {'epipolar','disparity','ncc','sift','BNB_NCC','BNB_SIFT'};

% Plot each ambiguity stage


% Plot each filter distribution
for i = 1:length(filters)
    filter_name = filters{i};
    filename = fullfile(output_dir, sprintf('%s_frame_%d.txt', filter_name, frame_idx));
    if exist(filename, 'file')
        fprintf('\n========================================\n');
        fprintf('Plotting filter: %s\n', filter_name);
        fprintf('========================================\n');
        plot_distribution(filter_name, frame_idx, output_dir);
    else
        fprintf('Warning: Filter file not found: %s\n', filename);
    end
end

fprintf('\n========================================\n');
fprintf('Done! All distributions plotted.\n');
fprintf('========================================\n');