% Visualize proximity histogram and cumulative density function
% Shows distribution of distances between KF edge locations and projected CF locations
% Helps determine optimal spatial grid search radius

clear; close all; clc;

%% Configuration
left_file = fullfile('..', 'outputs', 'proximity_distances_left.txt');
right_file = fullfile('..', 'outputs', 'proximity_distances_right.txt');

%% Load data
fprintf('Loading proximity distance data...\n');

% Read files, skipping comment lines starting with '#'
fid = fopen(left_file, 'r');
if fid == -1
    error('Could not open %s', left_file);
end
left_distances = [];
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line) && line(1) ~= '#'
        left_distances(end+1) = str2double(line);
    end
end
fclose(fid);

fid = fopen(right_file, 'r');
if fid == -1
    error('Could not open %s', right_file);
end
right_distances = [];
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line) && line(1) ~= '#'
        right_distances(end+1) = str2double(line);
    end
end
fclose(fid);

fprintf('Loaded %d left distances, %d right distances\n', ...
    length(left_distances), length(right_distances));

%% Statistics
fprintf('\n=== Left Camera Statistics ===\n');
fprintf('Mean:   %.2f px\n', mean(left_distances));
fprintf('Median: %.2f px\n', median(left_distances));
fprintf('Std:    %.2f px\n', std(left_distances));
fprintf('Min:    %.2f px\n', min(left_distances));
fprintf('Max:    %.2f px\n', max(left_distances));

fprintf('\n=== Right Camera Statistics ===\n');
fprintf('Mean:   %.2f px\n', mean(right_distances));
fprintf('Median: %.2f px\n', median(right_distances));
fprintf('Std:    %.2f px\n', std(right_distances));
fprintf('Min:    %.2f px\n', min(right_distances));
fprintf('Max:    %.2f px\n', max(right_distances));

%% Plot histogram
figure('Position', [100, 100, 1400, 600]);

subplot(2, 2, 1);
histogram(left_distances, 50, 'Normalization', 'probability');
xlabel('Distance (pixels)');
ylabel('Probability');
title('Left Camera: Histogram of KF→Projected CF Distance');
grid on;

subplot(2, 2, 2);
histogram(right_distances, 50, 'Normalization', 'probability');
xlabel('Distance (pixels)');
ylabel('Probability');
title('Right Camera: Histogram of KF→Projected CF Distance');
grid on;

%% Plot cumulative distribution function
subplot(2, 2, 3);
left_sorted = sort(left_distances);
left_cdf = (1:length(left_sorted)) / length(left_sorted);
plot(left_sorted, left_cdf, 'LineWidth', 2);
hold on;
% Mark percentiles
percentiles = [50, 90, 95, 99];
for p = percentiles
    val = prctile(left_distances, p);
    plot(val, p/100, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(val + 1, p/100, sprintf('%d%% = %.1fpx', p, val), ...
        'FontSize', 10, 'VerticalAlignment', 'middle');
end
% Mark common radii
radii = [10, 15, 20, 25, 30];
for r = radii
    coverage = sum(left_distances <= r) / length(left_distances) * 100;
    xline(r, '--', sprintf('r=%d (%.1f%%)', r, coverage), ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 9);
end
xlabel('Distance (pixels)');
ylabel('Cumulative Probability');
title('Left Camera: CDF of KF→Projected CF Distance');
grid on;
xlim([0, min(50, max(left_distances))]);
ylim([0, 1]);

subplot(2, 2, 4);
right_sorted = sort(right_distances);
right_cdf = (1:length(right_sorted)) / length(right_sorted);
plot(right_sorted, right_cdf, 'LineWidth', 2);
hold on;
% Mark percentiles
for p = percentiles
    val = prctile(right_distances, p);
    plot(val, p/100, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(val + 1, p/100, sprintf('%d%% = %.1fpx', p, val), ...
        'FontSize', 10, 'VerticalAlignment', 'middle');
end
% Mark common radii
for r = radii
    coverage = sum(right_distances <= r) / length(right_distances) * 100;
    xline(r, '--', sprintf('r=%d (%.1f%%)', r, coverage), ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 9);
end
xlabel('Distance (pixels)');
ylabel('Cumulative Probability');
title('Right Camera: CDF of KF→Projected CF Distance');
grid on;
xlim([0, min(50, max(right_distances))]);
ylim([0, 1]);

%% Print recommended radii
fprintf('\n=== Recommended Search Radii (for desired coverage) ===\n');
coverages = [90, 95, 99, 99.5, 99.9];
fprintf('Coverage | Left Radius | Right Radius\n');
fprintf('---------|-------------|-------------\n');
for c = coverages
    left_r = prctile(left_distances, c);
    right_r = prctile(right_distances, c);
    fprintf('%6.1f%% | %10.1f px | %11.1f px\n', c, left_r, right_r);
end

fprintf('\n=== Coverage for Common Radii ===\n');
fprintf('Radius | Left Coverage | Right Coverage\n');
fprintf('-------|---------------|---------------\n');
for r = [10, 15, 20, 25, 30]
    left_cov = sum(left_distances <= r) / length(left_distances) * 100;
    right_cov = sum(right_distances <= r) / length(right_distances) * 100;
    fprintf('%5d px | %12.1f%% | %13.1f%%\n', r, left_cov, right_cov);
end
