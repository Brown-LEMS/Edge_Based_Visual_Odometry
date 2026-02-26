% filepath: test/plot_all_distributions.m
% PLOT_ALL_DISTRIBUTIONS - Plot all filter distributions for a given frame
%
% This script plots PDF and CDF for all filters:
% - epipolar
% - location
% - ncc
% - sift
% - orientation

close all;
clear;
clc;

% Configuration
frame_idx = 0;  % Change this to plot different frames
output_dir = '../output_files';  % Root output directory

% List of filters to plot
filters = {'location'};

fprintf('\n========================================\n');
fprintf('Plotting all distributions for Frame %d\n', frame_idx);
fprintf('========================================\n');

% Plot each filter distribution
for i = 1:length(filters)
    filter_name = filters{i};
    
    % Check if the subdirectory and file exist
    filter_dir = fullfile(output_dir, 'values', filter_name);
    filename = fullfile(filter_dir, sprintf('frame_%d.txt', frame_idx));
    
    if exist(filename, 'file')
        fprintf('\n--- Plotting filter: %s ---\n', filter_name);
        try
            plot_distribution(filter_name, frame_idx, output_dir);
            fprintf('✓ Successfully plotted %s\n', filter_name);
        catch ME
            fprintf('✗ Error plotting %s: %s\n', filter_name, ME.message);
        end
    else
        fprintf('⚠ Warning: Filter file not found: %s\n', filename);
    end
end

fprintf('\n========================================\n');
fprintf('Done! All distributions plotted.\n');
fprintf('Output location: %s/values/{filter_name}/\n', output_dir);
fprintf('========================================\n');