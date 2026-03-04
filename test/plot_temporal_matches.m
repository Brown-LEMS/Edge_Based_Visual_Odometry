% filepath: test/plot_temporal_matches.m
function plot_temporal_matches(filename)
    % PLOT_TEMPORAL_MATCHES Parse temporal_matches.txt and create visualizations
    % Creates publication-ready figures showing temporal match statistics
    %
    % Args:
    %   filename: Path to temporal_matches.txt file (default: '../outputs/temporal_matches.txt')
    
    if nargin < 1
        filename = '../outputs/temporal_matches.txt';
    end
    
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    % Read and parse the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    kf_indices = [];
    cf_indices = [];
    left_matches = [];
    right_matches = [];
    
    line_count = 0;
    
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        
        line_count = line_count + 1;
        
        % Parse lines like: "(KF:0, CF:1), left temporal matches: 7935, right temporal matches: 7935"
        tokens = regexp(line, '\(KF:(\d+),\s*CF:(\d+)\),\s*left temporal matches:\s*(\d+),\s*right temporal matches:\s*(\d+)', 'tokens');
        if ~isempty(tokens)
            kf = str2double(tokens{1}{1});
            cf = str2double(tokens{1}{2});
            left = str2double(tokens{1}{3});
            right = str2double(tokens{1}{4});
            
            kf_indices = [kf_indices; kf];
            cf_indices = [cf_indices; cf];
            left_matches = [left_matches; left];
            right_matches = [right_matches; right];
        end
    end
    
    fclose(fid);
    
    % Display parsing summary
    fprintf('\n=== Temporal Matches Parsing Summary ===\n');
    fprintf('Total lines read: %d\n', line_count);
    fprintf('Frame pairs parsed: %d\n', length(kf_indices));
    fprintf('Left matches: n=%d, mean=%.1f, range=[%d, %d]\n', ...
        length(left_matches), mean(left_matches), min(left_matches), max(left_matches));
    fprintf('Right matches: n=%d, mean=%.1f, range=[%d, %d]\n', ...
        length(right_matches), mean(right_matches), min(right_matches), max(right_matches));
    
    % Get output directory from filename
    [output_dir, ~, ~] = fileparts(filename);
    if isempty(output_dir)
        output_dir = '.';
    end
    
    % Create visualizations
    plot_temporal_matches_over_time(kf_indices, cf_indices, left_matches, right_matches, output_dir);
    plot_temporal_matches_distribution(left_matches, 'Temporal Matches', [0.3, 0.5, 0.8], output_dir, 'temporal_matches');
    plot_temporal_matches_comparison(left_matches, right_matches, output_dir);
    
    fprintf('\n=== All plots saved to: %s ===\n', output_dir);
end

function plot_temporal_matches_over_time(kf_indices, cf_indices, left_matches, right_matches, output_dir)
    % PLOT_TEMPORAL_MATCHES_OVER_TIME Plot matches vs frame pairs
    
    fig = figure('Position', [100, 100, 1400, 800]);
    hold on;
    
    % Create x-axis as frame pair indices
    frame_pairs = 0:(length(kf_indices)-1);
    
    % Plot both left and right matches
    plot(frame_pairs, left_matches, '-o', 'LineWidth', 2.5, 'MarkerSize', 6, ...
        'Color', [0.2, 0.4, 0.8], 'MarkerFaceColor', [0.2, 0.4, 0.8], ...
        'DisplayName', 'Left Matches');
    plot(frame_pairs, right_matches, '-s', 'LineWidth', 2.5, 'MarkerSize', 6, ...
        'Color', [0.8, 0.2, 0.2], 'MarkerFaceColor', [0.8, 0.2, 0.2], ...
        'DisplayName', 'Right Matches');
    
    % Add mean lines
    mean_left = mean(left_matches);
    mean_right = mean(right_matches);
    plot([frame_pairs(1), frame_pairs(end)], [mean_left, mean_left], '--', ...
        'Color', [0.2, 0.4, 0.8], 'LineWidth', 2, 'DisplayName', sprintf('Left Mean (%.0f)', mean_left));
    plot([frame_pairs(1), frame_pairs(end)], [mean_right, mean_right], '--', ...
        'Color', [0.8, 0.2, 0.2], 'LineWidth', 2, 'DisplayName', sprintf('Right Mean (%.0f)', mean_right));
    
    % Styling
    xlabel('Frame Pair Index (KF to CF)', 'FontSize', 28, 'FontWeight', 'bold');
    ylabel('Number of Temporal Matches', 'FontSize', 28, 'FontWeight', 'bold');
    title('Temporal Matches Across Frame Pairs', 'FontSize', 32, 'FontWeight', 'bold');
    
    legend('Location', 'best', 'FontSize', 22);
    grid on;
    set(gca, 'FontSize', 24, 'LineWidth', 2);
    
    hold off;
    
    % Save figure
    save_filename = fullfile(output_dir, 'temporal_matches_over_time.png');
    saveas(fig, save_filename);
    fprintf('Saved temporal matches over time plot to: %s\n', save_filename);
    close(fig);
end

function plot_temporal_matches_distribution(match_counts, match_type, color, output_dir, filename_base)
    % PLOT_TEMPORAL_MATCHES_DISTRIBUTION Create PDF and CDF plots
    
    % Compute statistics
    mean_val = mean(match_counts);
    median_val = median(match_counts);
    std_val = std(match_counts);
    min_val = min(match_counts);
    max_val = max(match_counts);
    
    fprintf('\n=== %s Statistics ===\n', match_type);
    fprintf('  n=%d, μ=%.1f, σ=%.1f\n', length(match_counts), mean_val, std_val);
    fprintf('  Min: %d, Max: %d, Median: %.1f\n', min_val, max_val, median_val);
    
    %% --- PLOT 1: PDF ---
    fig_pdf = figure('Position', [100, 100, 1200, 900]);
    ax_main = axes('Parent', fig_pdf);
    hold(ax_main, 'on');
    
    % Define bin edges
    num_bins = min(50, ceil(sqrt(length(match_counts))));
    bin_edges = linspace(min_val, max_val, num_bins + 1);
    
    % Plot histogram
    histogram(ax_main, match_counts, bin_edges, 'Normalization', 'pdf', ...
        'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    
    % Add vertical lines for mean and median
    yl = ylim(ax_main);
    plot(ax_main, [mean_val, mean_val], yl, '--', 'Color', [0.9, 0.3, 0.3], ...
        'LineWidth', 4, 'DisplayName', 'Mean');
    plot(ax_main, [median_val, median_val], yl, '--', 'Color', [0.3, 0.9, 0.3], ...
        'LineWidth', 4, 'DisplayName', 'Median');
    
    % Styling
    xlabel(ax_main, 'Number of Temporal Matches', 'FontSize', 32, 'FontWeight', 'bold');
    ylabel(ax_main, 'Probability Density', 'FontSize', 32, 'FontWeight', 'bold');
    title(ax_main, sprintf('PDF - %s', match_type), 'FontSize', 36, 'FontWeight', 'bold');
    
    % Legend
    leg = legend(ax_main, 'Location', 'northeast', 'FontSize', 28);
    
    grid(ax_main, 'on');
    set(ax_main, 'FontSize', 28, 'LineWidth', 2);
    ax_main.XAxis.LineWidth = 2.5;
    ax_main.YAxis.LineWidth = 2.5;
    
    % Add statistics text box (positioned below the legend)
    text_str = sprintf('n=%d, μ=%.1f, σ=%.1f\nMedian=%.1f\nMin=%d, Max=%d', ...
        length(match_counts), mean_val, std_val, median_val, min_val, max_val);
    text(ax_main, 0.98, 0.60, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
        'FontSize', 24, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'Interpreter', 'none', 'LineWidth', 2);
    
    hold(ax_main, 'off');
    
    % Save PDF figure
    save_filename_pdf = fullfile(output_dir, sprintf('%s_pdf.png', filename_base));
    saveas(fig_pdf, save_filename_pdf);
    fprintf('Saved PDF figure to: %s\n', save_filename_pdf);
    close(fig_pdf);
    
    %% --- PLOT 2: CDF ---
    fig_cdf = figure('Position', [150, 150, 1200, 900]);
    ax_cdf = axes('Parent', fig_cdf);
    hold(ax_cdf, 'on');
    
    % Calculate ECDF
    [f, x] = ecdf(match_counts);
    
    % Plot CDF
    plot(ax_cdf, x, f*100, 'LineWidth', 7, 'Color', color);
    
    % Calculate percentiles
    percentiles = [50, 90, 95, 99];
    p_vals = prctile(match_counts, percentiles);
    
    % Plot percentile markers
    colors_percentile = [0.3, 0.9, 0.3;   % 50% - green
                        0.9, 0.7, 0.2;    % 90% - orange
                        0.9, 0.4, 0.2;    % 95% - red-orange
                        0.9, 0.2, 0.2];   % 99% - red
    
    for i = 1:length(percentiles)
        % Vertical line from x-axis to CDF
        plot(ax_cdf, [p_vals(i), p_vals(i)], [0, percentiles(i)], '--', ...
            'Color', colors_percentile(i,:), 'LineWidth', 3.5);
        
        % Horizontal line from y-axis to CDF
        plot(ax_cdf, [min_val, p_vals(i)], [percentiles(i), percentiles(i)], '--', ...
            'Color', colors_percentile(i,:), 'LineWidth', 3.5);
        
        % Mark point on CDF
        plot(ax_cdf, p_vals(i), percentiles(i), 'o', ...
            'MarkerSize', 12, 'MarkerFaceColor', colors_percentile(i,:), ...
            'MarkerEdgeColor', 'k', 'LineWidth', 2);
    end
    
    % Styling
    xlabel(ax_cdf, 'Number of Temporal Matches', 'FontSize', 36, 'FontWeight', 'bold');
    ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 36, 'FontWeight', 'bold');
    title(ax_cdf, sprintf('CDF - %s', match_type), 'FontSize', 38, 'FontWeight', 'bold');
    
    grid(ax_cdf, 'on');
    set(ax_cdf, 'FontSize', 32, 'LineWidth', 3);
    ax_cdf.XAxis.LineWidth = 3;
    ax_cdf.YAxis.LineWidth = 3;
    ylim(ax_cdf, [0, 100]);
    
    % Add percentile text annotations
    xl = xlim(ax_cdf);
    xr = xl(2) - xl(1);
    for i = 1:length(percentiles)
        % Position text slightly to the right of the percentile value
        text_x = p_vals(i) + xr * 0.02;
        text_y = percentiles(i) - 3;
        
        text(ax_cdf, text_x, text_y, sprintf('%d%%: %.0f', percentiles(i), p_vals(i)), ...
            'FontSize', 22, 'FontWeight', 'bold', 'Color', colors_percentile(i,:), ...
            'BackgroundColor', 'white', 'EdgeColor', colors_percentile(i,:), ...
            'LineWidth', 1.5, 'Margin', 3);
    end
    
    % Add statistics box
    stats_text = sprintf('n=%d\nμ=%.1f\nσ=%.1f\nMedian=%.1f', ...
        length(match_counts), mean_val, std_val, median_val);
    text(ax_cdf, 0.02, 0.98, stats_text, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontSize', 24, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'LineWidth', 2);
    
    hold(ax_cdf, 'off');
    
    % Save CDF figure
    save_filename_cdf = fullfile(output_dir, sprintf('%s_cdf.png', filename_base));
    saveas(fig_cdf, save_filename_cdf);
    fprintf('Saved CDF figure to: %s\n', save_filename_cdf);
    close(fig_cdf);
end

function plot_temporal_matches_comparison(left_matches, right_matches, output_dir)
    % PLOT_TEMPORAL_MATCHES_COMPARISON Create scatter and difference plots
    
    %% --- PLOT 1: Scatter Plot ---
    fig_scatter = figure('Position', [100, 100, 900, 900]);
    hold on;
    
    % Scatter plot
    scatter(left_matches, right_matches, 80, 'filled', 'MarkerFaceAlpha', 0.6, ...
        'MarkerFaceColor', [0.3, 0.5, 0.8]);
    
    % Add diagonal line (perfect agreement)
    min_val = min([min(left_matches), min(right_matches)]);
    max_val = max([max(left_matches), max(right_matches)]);
    plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 3, ...
        'DisplayName', 'Perfect Agreement');
    
    % Styling
    xlabel('Left Temporal Matches', 'FontSize', 28, 'FontWeight', 'bold');
    ylabel('Right Temporal Matches', 'FontSize', 28, 'FontWeight', 'bold');
    title('Left vs Right Temporal Matches', 'FontSize', 32, 'FontWeight', 'bold');
    
    legend('Location', 'southeast', 'FontSize', 22);
    grid on;
    set(gca, 'FontSize', 24, 'LineWidth', 2);
    axis equal;
    xlim([min_val, max_val]);
    ylim([min_val, max_val]);
    
    hold off;
    
    % Save scatter plot
    save_filename_scatter = fullfile(output_dir, 'temporal_matches_left_vs_right.png');
    saveas(fig_scatter, save_filename_scatter);
    fprintf('Saved left vs right scatter plot to: %s\n', save_filename_scatter);
    close(fig_scatter);
    
    %% --- PLOT 2: Difference Plot ---
    fig_diff = figure('Position', [150, 150, 1400, 800]);
    
    % Calculate differences
    differences = left_matches - right_matches;
    frame_pairs = 0:(length(left_matches)-1);
    
    hold on;
    
    % Bar plot of differences
    bar(frame_pairs, differences, 'FaceColor', [0.5, 0.5, 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    
    % Add zero line
    plot([frame_pairs(1), frame_pairs(end)], [0, 0], 'r-', 'LineWidth', 2);
    
    % Styling
    xlabel('Frame Pair Index (KF to CF)', 'FontSize', 28, 'FontWeight', 'bold');
    ylabel('Difference (Left - Right)', 'FontSize', 28, 'FontWeight', 'bold');
    title('Difference Between Left and Right Temporal Matches', 'FontSize', 32, 'FontWeight', 'bold');
    
    grid on;
    set(gca, 'FontSize', 24, 'LineWidth', 2);
    
    % Add statistics text box
    mean_diff = mean(differences);
    std_diff = std(differences);
    max_diff = max(abs(differences));
    text_str = sprintf('Mean Diff: %.2f\nStd Diff: %.2f\nMax |Diff|: %d', mean_diff, std_diff, max_diff);
    text(0.02, 0.98, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontSize', 22, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'LineWidth', 2);
    
    hold off;
    
    % Save difference plot
    save_filename_diff = fullfile(output_dir, 'temporal_matches_difference.png');
    saveas(fig_diff, save_filename_diff);
    fprintf('Saved difference plot to: %s\n', save_filename_diff);
    close(fig_diff);
end
