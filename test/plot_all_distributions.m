% filepath: test/plot_all_distributions.m
% PLOT_ALL_DISTRIBUTIONS - Plot all filter distributions for a given frame
%
% This script plots PDF and CDF for all filters:
% Stereo filters:
%   - epipolar
%   - location
%   - ncc
%   - sift
%   - orientation
% Temporal filters (combined left and right):
%   - temporal_location_error
%   - temporal_sift_distance
%   - temporal_ncc_score
%   - temporal_orientation_difference

close all;
clear;
clc;

% Configuration
frame_idx = 0;  % Change this to plot different frames
stereo_output_dir = '../output_files';  % Directory for stereo filters
temporal_output_dir = '../outputs';  % Directory for temporal filters

% List of filters to plot
% Stereo filters
stereo_filters = { 'orientation','ncc'};

% Temporal filters (base names - will plot left and right together)
temporal_filters = {
    'temporal_location_error_left', ...
    'temporal_sift_distance_left', ...
    'temporal_ncc_score_left', ...
    'temporal_orientation_difference_left'
};

% Combine all filters
filters = [stereo_filters, temporal_filters];

fprintf('\n========================================\n');
fprintf('Plotting all distributions for Frame %d\n', frame_idx);
fprintf('========================================\n');

% Counters
plotted_count = 0;
missing_count = 0;
error_count = 0;

fprintf('\n--- STEREO FILTERS ---\n');
for i = 1:length(stereo_filters)
    filter_name = stereo_filters{i};
    
    % Check if the subdirectory and file exist
    filter_dir = fullfile(stereo_output_dir, 'values', filter_name);
    filename = fullfile(filter_dir, sprintf('frame_%d.txt', frame_idx));
    
    if exist(filename, 'file')
        fprintf('\n--- Plotting filter: %s ---\n', filter_name);
        try
            plot_distribution(filter_name, frame_idx, stereo_output_dir);
            fprintf('✓ Successfully plotted %s\n', filter_name);
            plotted_count = plotted_count + 1;
        catch ME
            fprintf('✗ Error plotting %s: %s\n', filter_name, ME.message);
            error_count = error_count + 1;
        end
    else
        fprintf('⚠ Warning: Filter file not found: %s\n', filename);
        missing_count = missing_count + 1;
    end
end




fprintf('\n========================================\n');
fprintf('Summary:\n');
fprintf('  ✓ Successfully plotted: %d\n', plotted_count);
fprintf('  ⚠ Files not found: %d\n', missing_count);
fprintf('  ✗ Errors: %d\n', error_count);
fprintf('\nStereo output: %s/values/{filter_name}/\n', stereo_output_dir);
fprintf('Temporal output: %s/values/{filter_name}/\n', temporal_output_dir);
fprintf('========================================\n');