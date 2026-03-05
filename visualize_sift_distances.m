% Visualize SIFT distances from debug file
% File format: edge_idx candidate_idx sift_distance passed(1/0)

clear; close all;

% Load data (skip header line starting with #)
fid = fopen('sift_distances_debug.txt', 'r');
if fid == -1
    error('Cannot open sift_distances_debug.txt');
end
% Skip header line
fgetl(fid);
% Read numeric data
data = fscanf(fid, '%d %d %f %d', [4 inf])';
fclose(fid);
edge_idx = data(:,1);
candidate_idx = data(:,2);
sift_dist = data(:,3);
passed = data(:,4);

fprintf('=== SIFT Distance Statistics ===\n');
fprintf('Total comparisons: %d\n', length(sift_dist));
fprintf('Passed (dist < 200): %d (%.1f%%)\n', sum(passed), 100*sum(passed)/length(passed));
fprintf('Failed (dist >= 200): %d (%.1f%%)\n', sum(~passed), 100*sum(~passed)/length(passed));
fprintf('\nDistance statistics:\n');
fprintf('  Min:    %.2f\n', min(sift_dist));
fprintf('  Max:    %.2f\n', max(sift_dist));
fprintf('  Mean:   %.2f\n', mean(sift_dist));
fprintf('  Median: %.2f\n', median(sift_dist));
fprintf('  Std:    %.2f\n', std(sift_dist));

% Per-edge statistics
unique_edges = unique(edge_idx);
num_edges = length(unique_edges);
edges_with_passed = 0;
edges_with_failed_only = 0;
failed_only_min_dists = [];

for i = 1:num_edges
    edge_mask = edge_idx == unique_edges(i);
    edge_passed = sum(passed(edge_mask));
    if edge_passed > 0
        edges_with_passed = edges_with_passed + 1;
    else
        edges_with_failed_only = edges_with_failed_only + 1;
        % Record the best (minimum) distance for this edge that had no passes
        failed_only_min_dists(end+1) = min(sift_dist(edge_mask));
    end
end

fprintf('\n=== Per-Edge Statistics ===\n');
fprintf('Total edges: %d\n', num_edges);
fprintf('Edges with at least 1 passing candidate: %d (%.1f%%)\n', edges_with_passed, 100*edges_with_passed/num_edges);
fprintf('Edges with only failing candidates: %d (%.1f%%)\n', edges_with_failed_only, 100*edges_with_failed_only/num_edges);
fprintf('\nFor edges with NO passing candidates:\n');
fprintf('  Best distance - Min: %.2f, Max: %.2f, Mean: %.2f, Median: %.2f\n', ...
    min(failed_only_min_dists), max(failed_only_min_dists), mean(failed_only_min_dists), median(failed_only_min_dists));
fprintf('  Within 50 of threshold (200-250): %d (%.1f%%)\n', ...
    sum(failed_only_min_dists >= 200 & failed_only_min_dists < 250), ...
    100*sum(failed_only_min_dists >= 200 & failed_only_min_dists < 250)/length(failed_only_min_dists));

%% Figure 1: Overall distribution
figure('Position', [100 100 1600 900]);

% Subplot 1: Histogram of all distances
subplot(2,4,1);
histogram(sift_dist, 100, 'FaceColor', [0.3 0.3 0.8]);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold=200');
xlabel('SIFT Distance');
ylabel('Count');
title('Distribution of All SIFT Distances');
grid on;

% Subplot 2: Histogram of passed vs failed
subplot(2,4,2);
histogram(sift_dist(passed==1), 50, 'FaceColor', [0.2 0.8 0.2], 'FaceAlpha', 0.7);
hold on;
histogram(sift_dist(passed==0), 50, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.7);
xline(200, 'k--', 'LineWidth', 2);
xlabel('SIFT Distance');
ylabel('Count');
title('Passed (green) vs Failed (red)');
legend('Passed', 'Failed', 'Threshold');
grid on;

% Subplot 3: Best distances for edges with NO passing candidates
subplot(2,4,3);
histogram(failed_only_min_dists, 50, 'FaceColor', [0.8 0.4 0.4]);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold');
xlabel('Best SIFT Distance');
ylabel('Number of Edges');
title(sprintf('Edges with No Passes (%d edges)', length(failed_only_min_dists)));
grid on;
legend('Best distance', 'Threshold');

% Subplot 4: Cumulative distribution
subplot(2,4,4);
sorted_dist = sort(sift_dist);
cdf_vals = (1:length(sorted_dist))' / length(sorted_dist);
plot(sorted_dist, cdf_vals, 'b-', 'LineWidth', 2);
hold on;
xline(200, 'r--', 'LineWidth', 2);
yline(sum(passed)/length(passed), 'g--', 'LineWidth', 1.5);
xlabel('SIFT Distance');
ylabel('CDF');
title('Cumulative Distribution');
grid on;
legend('CDF', 'Threshold=200', sprintf('%.1f%% passed', 100*sum(passed)/length(passed)));

% Subplot 5: Distance near threshold
subplot(2,4,5);
near_threshold = sift_dist >= 150 & sift_dist <= 250;
histogram(sift_dist(near_threshold), 50, 'FaceColor', [0.8 0.5 0.2]);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold');
xlabel('SIFT Distance');
ylabel('Count');
title('Zoomed: Distances near Threshold [150-250]');
grid on;

% Subplot 6: Per-edge candidate counts
subplot(2,4,6);
candidates_per_edge = arrayfun(@(e) sum(edge_idx == e), unique_edges);
passed_per_edge = arrayfun(@(e) sum(edge_idx == e & passed == 1), unique_edges);
histogram(candidates_per_edge, 50, 'FaceColor', [0.5 0.5 0.8]);
hold on;
histogram(passed_per_edge, 50, 'FaceColor', [0.2 0.8 0.2]);
xlabel('Number of Candidates');
ylabel('Number of Edges');
title('Candidates per Edge');
legend('Total candidates', 'Passed candidates');
grid on;

% Subplot 7: Box plot comparison
subplot(2,4,7);
boxplot([sift_dist(passed==1); sift_dist(passed==0)], ...
        [ones(sum(passed),1); 2*ones(sum(~passed),1)], ...
        'Labels', {'Passed', 'Failed'}, 'Colors', [0.2 0.8 0.2; 0.8 0.2 0.2]);
hold on;
yline(200, 'r--', 'LineWidth', 2);
ylabel('SIFT Distance');
title('Distance Distribution: Passed vs Failed');
grid on;

% Subplot 8: Failed-only edges near threshold
subplot(2,4,8);
near_thresh_failed = failed_only_min_dists >= 180 & failed_only_min_dists <= 220;
histogram(failed_only_min_dists(near_thresh_failed), 30, 'FaceColor', [0.9 0.3 0.3]);
hold on;
xline(200, 'r--', 'LineWidth', 2);
xlabel('Best SIFT Distance');
ylabel('Number of Edges');
title(sprintf('Failed Edges Near Threshold [180-220]\n(%d edges)', sum(near_thresh_failed)));
grid on;

sgtitle('SIFT Distance Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Figure 2: Per-edge analysis
figure('Position', [150 150 1400 600]);

% Average distance per edge
subplot(1,3,1);
avg_dist_per_edge = arrayfun(@(e) mean(sift_dist(edge_idx == e)), unique_edges);
plot(avg_dist_per_edge, 'b.', 'MarkerSize', 4);
hold on;
yline(200, 'r--', 'LineWidth', 2);
xlabel('Edge Index');
ylabel('Average SIFT Distance');
title('Average Distance per Edge');
grid on;

% Min distance per edge
subplot(1,3,2);
min_dist_per_edge = arrayfun(@(e) min(sift_dist(edge_idx == e)), unique_edges);
plot(min_dist_per_edge, 'g.', 'MarkerSize', 4);
hold on;
yline(200, 'r--', 'LineWidth', 2);
xlabel('Edge Index');
ylabel('Min SIFT Distance');
title('Best Match Distance per Edge');
grid on;

% Histogram of min distances
subplot(1,3,3);
histogram(min_dist_per_edge, 50, 'FaceColor', [0.3 0.7 0.3]);
hold on;
xline(200, 'r--', 'LineWidth', 2);
xlabel('Min SIFT Distance');
ylabel('Number of Edges');
title('Distribution of Best Match Distance per Edge');
grid on;

sgtitle('Per-Edge SIFT Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Additional statistics
fprintf('\n=== Distance Range Analysis ===\n');
ranges = [0 50; 50 100; 100 150; 150 200; 200 250; 250 300; 300 inf];
for i = 1:size(ranges, 1)
    count = sum(sift_dist >= ranges(i,1) & sift_dist < ranges(i,2));
    if ranges(i,2) == inf
        fprintf('[%d, inf): %d (%.1f%%)\n', ranges(i,1), count, 100*count/length(sift_dist));
    else
        fprintf('[%d, %d): %d (%.1f%%)\n', ranges(i,1), ranges(i,2), count, 100*count/length(sift_dist));
    end
end

fprintf('\n=== Threshold Sensitivity ===\n');
thresholds = [150, 175, 200, 225, 250, 300, 350, 400];
for thresh = thresholds
    would_pass = sum(sift_dist < thresh);
    edges_with_match = length(unique(edge_idx(sift_dist < thresh)));
    fprintf('Threshold=%d: %d pairs (%.1f%%), %d edges (%.1f%%)\n', ...
        thresh, would_pass, 100*would_pass/length(sift_dist), ...
        edges_with_match, 100*edges_with_match/num_edges);
end

fprintf('\nDone! Figures saved.\n');
