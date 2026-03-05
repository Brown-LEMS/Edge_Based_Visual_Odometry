% Complete visualization of SIFT and NCC filtering for temporal matching
% Reads both sift_distances_debug.txt and ncc_debug.txt

%% Load SIFT data
fprintf('Loading SIFT debug data...\n');
fid_sift = fopen('sift_distances_debug.txt', 'r');
if fid_sift == -1
    error('Cannot open sift_distances_debug.txt');
end
fgetl(fid_sift); % Skip header
sift_data = fscanf(fid_sift, '%d %d %f %d', [4 Inf])';
fclose(fid_sift);

sift_edge_idx = sift_data(:,1);
sift_candidate_idx = sift_data(:,2);
sift_distances = sift_data(:,3);
sift_passed = sift_data(:,4);

fprintf('Loaded %d SIFT comparisons\n', length(sift_distances));

%% Load NCC data
fprintf('Loading NCC debug data...\n');
fid_ncc = fopen('ncc_debug.txt', 'r');
if fid_ncc == -1
    error('Cannot open ncc_debug.txt');
end
fgetl(fid_ncc); % Skip header
ncc_data = fscanf(fid_ncc, '%d %d %f %d', [4 Inf])';
fclose(fid_ncc);

ncc_edge_idx = ncc_data(:,1);
ncc_candidate_idx = ncc_data(:,2);
ncc_scores = ncc_data(:,3);
ncc_passed = ncc_data(:,4);

fprintf('Loaded %d NCC comparisons\n', length(ncc_scores));

%% SIFT Analysis
fprintf('\\n=== SIFT FILTERING ANALYSIS ===\\n');
sift_passed_dists = sift_distances(sift_passed == 1);
sift_failed_dists = sift_distances(sift_passed == 0);

fprintf('SIFT Overall Statistics:\\n');
fprintf('  Total comparisons: %d\\n', length(sift_distances));
fprintf('  Passed: %d (%.1f%%)\\n', length(sift_passed_dists), 100*length(sift_passed_dists)/length(sift_distances));
fprintf('  Failed: %d (%.1f%%)\\n', length(sift_failed_dists), 100*length(sift_failed_dists)/length(sift_distances));
fprintf('  Min distance: %.2f\\n', min(sift_distances));
fprintf('  Max distance: %.2f\\n', max(sift_distances));
fprintf('  Mean distance: %.2f\\n', mean(sift_distances));
fprintf('  Median distance: %.2f\\n', median(sift_distances));
fprintf('  Std dev: %.2f\\n', std(sift_distances));

% Per-edge SIFT analysis
unique_sift_edges = unique(sift_edge_idx);
num_edges_with_sift = length(unique_sift_edges);
edges_with_sift_matches = 0;
edges_without_sift_matches = 0;
sift_failed_only_min_dists = [];

for edge = unique_sift_edges'
    edge_mask = (sift_edge_idx == edge);
    edge_passed = sum(sift_passed(edge_mask));
    
    if edge_passed > 0
        edges_with_sift_matches = edges_with_sift_matches + 1;
    else
        edges_without_sift_matches = edges_without_sift_matches + 1;
        % Record the best (minimum) distance for this edge
        edge_dists = sift_distances(edge_mask);
        sift_failed_only_min_dists = [sift_failed_only_min_dists; min(edge_dists)];
    end
end

fprintf('\\nSIFT Per-Edge Statistics:\\n');
fprintf('  Edges with SIFT candidates: %d\\n', num_edges_with_sift);
fprintf('  Edges with passing matches: %d (%.1f%%)\\n', edges_with_sift_matches, 100*edges_with_sift_matches/num_edges_with_sift);
fprintf('  Edges with ZERO passing matches: %d (%.1f%%)\\n', edges_without_sift_matches, 100*edges_without_sift_matches/num_edges_with_sift);

if ~isempty(sift_failed_only_min_dists)
    fprintf('\\nSIFT Edges with NO passing candidates (best distances):\\n');
    fprintf('  Min best distance: %.2f\\n', min(sift_failed_only_min_dists));
    fprintf('  Max best distance: %.2f\\n', max(sift_failed_only_min_dists));
    fprintf('  Mean best distance: %.2f\\n', mean(sift_failed_only_min_dists));
    fprintf('  Median best distance: %.2f\\n', median(sift_failed_only_min_dists));
    fprintf('  Within 200-250: %d (%.1f%%)\\n', sum(sift_failed_only_min_dists >= 200 & sift_failed_only_min_dists <= 250), ...
            100*sum(sift_failed_only_min_dists >= 200 & sift_failed_only_min_dists <= 250)/length(sift_failed_only_min_dists));
end

%% NCC Analysis
fprintf('\\n=== NCC FILTERING ANALYSIS ===\\n');
ncc_passed_scores = ncc_scores(ncc_passed == 1);
ncc_failed_scores = ncc_scores(ncc_passed == 0);

fprintf('NCC Overall Statistics:\\n');
fprintf('  Total comparisons: %d\\n', length(ncc_scores));
fprintf('  Passed: %d (%.1f%%)\\n', length(ncc_passed_scores), 100*length(ncc_passed_scores)/length(ncc_scores));
fprintf('  Failed: %d (%.1f%%)\\n', length(ncc_failed_scores), 100*length(ncc_failed_scores)/length(ncc_scores));
fprintf('  Min NCC: %.4f\\n', min(ncc_scores));
fprintf('  Max NCC: %.4f\\n', max(ncc_scores));
fprintf('  Mean NCC: %.4f\\n', mean(ncc_scores));
fprintf('  Median NCC: %.4f\\n', median(ncc_scores));
fprintf('  Std dev: %.4f\\n', std(ncc_scores));

% Per-edge NCC analysis
unique_ncc_edges = unique(ncc_edge_idx);
num_edges_with_ncc = length(unique_ncc_edges);
edges_with_ncc_matches = 0;
edges_without_ncc_matches = 0;
ncc_failed_only_max_scores = [];

for edge = unique_ncc_edges'
    edge_mask = (ncc_edge_idx == edge);
    edge_passed = sum(ncc_passed(edge_mask));
    
    if edge_passed > 0
        edges_with_ncc_matches = edges_with_ncc_matches + 1;
    else
        edges_without_ncc_matches = edges_without_ncc_matches + 1;
        % Record the best (maximum) NCC score for this edge
        edge_scores = ncc_scores(edge_mask);
        ncc_failed_only_max_scores = [ncc_failed_only_max_scores; max(edge_scores)];
    end
end

fprintf('\\nNCC Per-Edge Statistics:\\n');
fprintf('  Edges with NCC candidates: %d\\n', num_edges_with_ncc);
fprintf('  Edges with passing matches: %d (%.1f%%)\\n', edges_with_ncc_matches, 100*edges_with_ncc_matches/num_edges_with_ncc);
fprintf('  Edges with ZERO passing matches: %d (%.1f%%)\\n', edges_without_ncc_matches, 100*edges_without_ncc_matches/num_edges_with_ncc);

if ~isempty(ncc_failed_only_max_scores)
    fprintf('\\nNCC Edges with NO passing candidates (best scores):\\n');
    fprintf('  Min best NCC: %.4f\\n', min(ncc_failed_only_max_scores));
    fprintf('  Max best NCC: %.4f\\n', max(ncc_failed_only_max_scores));
    fprintf('  Mean best NCC: %.4f\\n', mean(ncc_failed_only_max_scores));
    fprintf('  Median best NCC: %.4f\\n', median(ncc_failed_only_max_scores));
    fprintf('  Within 0.70-0.80: %d (%.1f%%)\\n', sum(ncc_failed_only_max_scores >= 0.70 & ncc_failed_only_max_scores <= 0.80), ...
            100*sum(ncc_failed_only_max_scores >= 0.70 & ncc_failed_only_max_scores <= 0.80)/length(ncc_failed_only_max_scores));
end

%% Threshold Sensitivity Analysis
fprintf('\\n=== THRESHOLD SENSITIVITY ===\\n');
sift_thresholds = [150, 175, 200, 225, 250, 300, 350, 400];
ncc_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

fprintf('\\nSIFT Threshold Sensitivity:\\n');
for thresh = sift_thresholds
    num_passing = sum(sift_distances < thresh);
    fprintf('  Threshold < %.0f: %d matches (%.1f%%)\\n', thresh, num_passing, 100*num_passing/length(sift_distances));
end

fprintf('\\nNCC Threshold Sensitivity:\\n');
for thresh = ncc_thresholds
    num_passing = sum(ncc_scores > thresh);
    fprintf('  Threshold > %.2f: %d matches (%.1f%%)\\n', thresh, num_passing, 100*num_passing/length(ncc_scores));
end

%% SIFT Visualization
figure('Position', [100, 100, 1800, 1200], 'Name', 'SIFT Filtering Analysis');

% 1. Overall SIFT distribution
subplot(3,3,1);
histogram(sift_distances, 100, 'FaceColor', [0.3 0.5 0.8]);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold=200');
xlabel('SIFT Distance');
ylabel('Count');
title(sprintf('SIFT Distance Distribution (n=%d)', length(sift_distances)));
grid on;

% 2. Passed vs Failed SIFT
subplot(3,3,2);
histogram(sift_passed_dists, 50, 'FaceColor', 'g', 'FaceAlpha', 0.6, 'DisplayName', 'Passed');
hold on;
histogram(sift_failed_dists, 50, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'DisplayName', 'Failed');
xline(200, 'k--', 'LineWidth', 2);
xlabel('SIFT Distance');
ylabel('Count');
title('Passed vs Failed SIFT');
legend;
grid on;

% 3. Best distances for edges with NO matches
subplot(3,3,3);
if ~isempty(sift_failed_only_min_dists)
    histogram(sift_failed_only_min_dists, 50, 'FaceColor', [0.8 0.3 0.3]);
    hold on;
    xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold=200');
    xlabel('Best SIFT Distance');
    ylabel('Count');
    title(sprintf('Zero-Match Edges Best Distances (n=%d)', length(sift_failed_only_min_dists)));
    grid on;
end

% 4. SIFT CDF
subplot(3,3,4);
[f, x] = ecdf(sift_distances);
plot(x, f*100, 'LineWidth', 2);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Threshold=200');
xlabel('SIFT Distance');
ylabel('Cumulative %');
title('SIFT Distance CDF');
grid on;

% 5. Near threshold zoom
subplot(3,3,5);
histogram(sift_distances(sift_distances >= 150 & sift_distances <= 250), 100, 'FaceColor', [0.5 0.5 0.8]);
hold on;
xline(200, 'r--', 'LineWidth', 2);
xlabel('SIFT Distance');
ylabel('Count');
title('Near Threshold [150-250]');
xlim([150 250]);
grid on;

% 6. Per-edge candidate counts
subplot(3,3,6);
edge_counts = histcounts(sift_edge_idx, length(unique_sift_edges));
histogram(edge_counts, 50, 'FaceColor', [0.6 0.4 0.7]);
xlabel('Candidates per Edge');
ylabel('Number of Edges');
title('SIFT Candidates per Edge');
grid on;

% 7. Box plot
subplot(3,3,7);
boxplot([sift_passed_dists; sift_failed_dists], [ones(size(sift_passed_dists)); 2*ones(size(sift_failed_dists))], ...
        'Labels', {'Passed', 'Failed'});
ylabel('SIFT Distance');
title('SIFT Distance Box Plot');
grid on;

% 8. Failed edges near threshold
subplot(3,3,8);
near_thresh_failed = sift_failed_dists(sift_failed_dists >= 180 & sift_failed_dists <= 220);
histogram(near_thresh_failed, 40, 'FaceColor', [0.9 0.4 0.4]);
hold on;
xline(200, 'r--', 'LineWidth', 2);
xlabel('SIFT Distance');
ylabel('Count');
title(sprintf('Failed Near Threshold [180-220] (n=%d)', length(near_thresh_failed)));
xlim([180 220]);
grid on;

% 9. Threshold sensitivity curve
subplot(3,3,9);
test_thresholds = 100:10:500;
passing_counts = arrayfun(@(t) sum(sift_distances < t), test_thresholds);
plot(test_thresholds, passing_counts/length(sift_distances)*100, 'b-', 'LineWidth', 2);
hold on;
xline(200, 'r--', 'LineWidth', 2, 'Label', 'Current=200');
xlabel('SIFT Threshold');
ylabel('% Passing');
title('SIFT Threshold Sensitivity');
grid on;

%% NCC Visualization
figure('Position', [150, 150, 1800, 1200], 'Name', 'NCC Filtering Analysis');

% 1. Overall NCC distribution
subplot(3,3,1);
histogram(ncc_scores, 100, 'FaceColor', [0.3 0.7 0.5]);
hold on;
xline(0.8, 'r--', 'LineWidth', 2, 'Label', 'Threshold=0.8');
xlabel('NCC Score');
ylabel('Count');
title(sprintf('NCC Score Distribution (n=%d)', length(ncc_scores)));
grid on;

% 2. Passed vs Failed NCC
subplot(3,3,2);
histogram(ncc_passed_scores, 50, 'FaceColor', 'g', 'FaceAlpha', 0.6, 'DisplayName', 'Passed');
hold on;
histogram(ncc_failed_scores, 50, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'DisplayName', 'Failed');
xline(0.8, 'k--', 'LineWidth', 2);
xlabel('NCC Score');
ylabel('Count');
title('Passed vs Failed NCC');
legend;
grid on;

% 3. Best scores for edges with NO matches
subplot(3,3,3);
if ~isempty(ncc_failed_only_max_scores)
    histogram(ncc_failed_only_max_scores, 50, 'FaceColor', [0.8 0.3 0.3]);
    hold on;
    xline(0.8, 'r--', 'LineWidth', 2, 'Label', 'Threshold=0.8');
    xlabel('Best NCC Score');
    ylabel('Count');
    title(sprintf('Zero-Match Edges Best Scores (n=%d)', length(ncc_failed_only_max_scores)));
    grid on;
end

% 4. NCC CDF
subplot(3,3,4);
[f, x] = ecdf(ncc_scores);
plot(x, f*100, 'LineWidth', 2);
hold on;
xline(0.8, 'r--', 'LineWidth', 2, 'Label', 'Threshold=0.8');
xlabel('NCC Score');
ylabel('Cumulative %');
title('NCC Score CDF');
grid on;

% 5. Near threshold zoom
subplot(3,3,5);
histogram(ncc_scores(ncc_scores >= 0.6 & ncc_scores <= 1.0), 100, 'FaceColor', [0.5 0.8 0.6]);
hold on;
xline(0.8, 'r--', 'LineWidth', 2);
xlabel('NCC Score');
ylabel('Count');
title('Near Threshold [0.6-1.0]');
xlim([0.6 1.0]);
grid on;

% 6. Per-edge candidate counts
subplot(3,3,6);
ncc_edge_counts = histcounts(ncc_edge_idx, length(unique_ncc_edges));
histogram(ncc_edge_counts, 50, 'FaceColor', [0.6 0.7 0.4]);
xlabel('Candidates per Edge');
ylabel('Number of Edges');
title('NCC Candidates per Edge');
grid on;

% 7. Box plot
subplot(3,3,7);
boxplot([ncc_passed_scores; ncc_failed_scores], [ones(size(ncc_passed_scores)); 2*ones(size(ncc_failed_scores))], ...
        'Labels', {'Passed', 'Failed'});
ylabel('NCC Score');
title('NCC Score Box Plot');
grid on;

% 8. Failed edges near threshold
subplot(3,3,8);
near_thresh_ncc_failed = ncc_failed_scores(ncc_failed_scores >= 0.7 & ncc_failed_scores <= 0.85);
histogram(near_thresh_ncc_failed, 40, 'FaceColor', [0.9 0.4 0.4]);
hold on;
xline(0.8, 'r--', 'LineWidth', 2);
xlabel('NCC Score');
ylabel('Count');
title(sprintf('Failed Near Threshold [0.7-0.85] (n=%d)', length(near_thresh_ncc_failed)));
xlim([0.7 0.85]);
grid on;

% 9. Threshold sensitivity curve
subplot(3,3,9);
test_ncc_thresholds = 0.5:0.02:1.0;
ncc_passing_counts = arrayfun(@(t) sum(ncc_scores > t), test_ncc_thresholds);
plot(test_ncc_thresholds, ncc_passing_counts/length(ncc_scores)*100, 'b-', 'LineWidth', 2);
hold on;
xline(0.8, 'r--', 'LineWidth', 2, 'Label', 'Current=0.8');
xlabel('NCC Threshold');
ylabel('% Passing');
title('NCC Threshold Sensitivity');
grid on;

%% Combined Pipeline View
figure('Position', [200, 200, 1400, 600], 'Name', 'Combined Filtering Pipeline');

% 1. SIFT and NCC joint distribution (for matched edges)
subplot(1,3,1);
scatter(sift_distances(sift_passed == 1), ncc_scores(ncc_passed == 1), 20, 'g', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
scatter(sift_distances(sift_passed == 0), ncc_scores(ncc_passed == 0), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.1);
xline(200, 'b--', 'LineWidth', 2);
yline(0.8, 'b--', 'LineWidth', 2);
xlabel('SIFT Distance');
ylabel('NCC Score');
title('SIFT vs NCC (Green=Both Pass, Red=At Least One Fail)');
grid on;

% 2. Filtering cascade
subplot(1,3,2);
stages = {'Initial', 'Post-SIFT', 'Post-NCC'};
counts = [length(sift_distances), length(sift_passed_dists), length(ncc_passed_scores)];
bar(counts, 'FaceColor', [0.4 0.6 0.8]);
set(gca, 'XTickLabel', stages);
ylabel('Number of Matches');
title('Filtering Cascade');
grid on;
for i = 1:length(counts)
    text(i, counts(i)+1000, sprintf('%d', counts(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

% 3. Loss breakdown
subplot(1,3,3);
sift_loss = length(sift_distances) - length(sift_passed_dists);
ncc_loss = length(sift_passed_dists) - length(ncc_passed_scores);
final_kept = length(ncc_passed_scores);
data = [final_kept; sift_loss; ncc_loss];
labels = {'Final Kept', 'Lost in SIFT', 'Lost in NCC'};
pie(data, labels);
title(sprintf('Match Loss Breakdown (Final: %d)', final_kept));

fprintf('\\n=== ANALYSIS COMPLETE ===\\n');
fprintf('Final matches after both filters: %d\\n', length(ncc_passed_scores));
fprintf('Loss from SIFT filtering: %d (%.1f%%)\\n', sift_loss, 100*sift_loss/length(sift_distances));
fprintf('Loss from NCC filtering: %d (%.1f%%)\\n', ncc_loss, 100*ncc_loss/length(sift_passed_dists));
