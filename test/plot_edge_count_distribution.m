% filepath: test/plot_edge_count_distribution.m
function plot_edge_count_distribution(filename)
    % PLOT_EDGE_COUNT_DISTRIBUTION Parse edge_numbers.txt and plot distributions
    % Creates 6 publication-ready figures (PDF and CDF for each edge type)
    %
    % Args:
    %   filename: Path to edge_numbers.txt file (default: '../output_files/edge_numbers.txt')
    
    if nargin < 1
        filename = '../output_files/edge_numbers.txt';
    end
    
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    % Read and parse the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    left_edge_counts = [];
    right_edge_counts = [];
    stereo_left_edge_counts = [];
    frame_indices = [];
    
    line_count = 0;
    left_line_count = 0;
    stereo_line_count = 0;
    
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        
        line_count = line_count + 1;
        
        % Parse lines like: "Frame:0, left edge numbers: 49875 right edge numbers: 48675"
        if contains(line, 'left edge numbers:') && contains(line, 'right edge numbers:')
            tokens = regexp(line, 'Frame:(\d+),\s*left edge numbers:\s*(\d+)\s*right edge numbers:\s*(\d+)', 'tokens');
            if ~isempty(tokens)
                frame_idx = str2double(tokens{1}{1});
                left_count = str2double(tokens{1}{2});
                right_count = str2double(tokens{1}{3});
                
                left_edge_counts = [left_edge_counts; left_count];
                right_edge_counts = [right_edge_counts; right_count];
                frame_indices = [frame_indices; frame_idx];
                left_line_count = left_line_count + 1;
            end
        % Parse lines like: "Frame:0, after stereo left edge numbers: 16292"
        elseif contains(line, 'after stereo left edge numbers:')
            tokens = regexp(line, 'Frame:(\d+),\s*after stereo left edge numbers:\s*(\d+)', 'tokens');
            if ~isempty(tokens)
                stereo_count = str2double(tokens{1}{2});
                stereo_left_edge_counts = [stereo_left_edge_counts; stereo_count];
                stereo_line_count = stereo_line_count + 1;
            end
        end
    end
    
    fclose(fid);
    
    fprintf('\n=== Parsing Summary ===\n');
    fprintf('Total lines read: %d\n', line_count);
    fprintf('Left edge lines parsed: %d\n', left_line_count);
    fprintf('Stereo edge lines parsed: %d\n', stereo_line_count);
    fprintf('Loaded data from %d frames\n', length(frame_indices));
    fprintf('Left edges: n=%d, mean=%.1f, range=[%d, %d]\n', ...
        length(left_edge_counts), mean(left_edge_counts), min(left_edge_counts), max(left_edge_counts));
    fprintf('Right edges: n=%d, mean=%.1f, range=[%d, %d]\n', ...
        length(right_edge_counts), mean(right_edge_counts), min(right_edge_counts), max(right_edge_counts));
    fprintf('Stereo edges: n=%d, mean=%.1f, range=[%d, %d]\n', ...
        length(stereo_left_edge_counts), mean(stereo_left_edge_counts), ...
        min(stereo_left_edge_counts), max(stereo_left_edge_counts));
    
    % Get output directory from filename
    [output_dir, ~, ~] = fileparts(filename);
    if isempty(output_dir)
        output_dir = '.';
    end
    
    % Plot each edge type with PDF and CDF
    plot_edge_type_distribution(left_edge_counts, 'Left Edges', [0.2, 0.4, 0.8], output_dir, 'left_edges');
    plot_edge_type_distribution(right_edge_counts, 'Right Edges', [0.8, 0.2, 0.2], output_dir, 'right_edges');
    plot_edge_type_distribution(stereo_left_edge_counts, 'Stereo Correspondence', [0.2, 0.7, 0.3], output_dir, 'stereo_correspondence');
end

function plot_edge_type_distribution(edge_counts, edge_type, color, output_dir, filename_base)
    % PLOT_EDGE_TYPE_DISTRIBUTION Create publication-ready PDF and CDF plots
    
    % Compute statistics
    mean_val = mean(edge_counts);
    median_val = median(edge_counts);
    std_val = std(edge_counts);
    min_val = min(edge_counts);
    max_val = max(edge_counts);
    
    fprintf('\n=== %s Statistics ===\n', edge_type);
    fprintf('  n=%d, μ=%.1f, σ=%.1f\n', length(edge_counts), mean_val, std_val);
    fprintf('  Min: %d, Max: %d, Median: %.1f\n', min_val, max_val, median_val);
    
    %% --- PLOT 1: PDF ---
    fig_pdf = figure('Position', [100, 100, 1200, 900]);
    ax_main = axes('Parent', fig_pdf);
    hold(ax_main, 'on');
    
    % Define bin edges
    num_bins = min(50, ceil(sqrt(length(edge_counts))));
    bin_edges = linspace(min_val, max_val, num_bins + 1);
    
    % Plot histogram
    histogram(ax_main, edge_counts, bin_edges, 'Normalization', 'pdf', ...
        'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    
    % Add vertical lines for mean and median
    yl = ylim(ax_main);
    plot(ax_main, [mean_val, mean_val], yl, '--', 'Color', [0.9, 0.3, 0.3], ...
        'LineWidth', 4, 'DisplayName', 'Mean');
    plot(ax_main, [median_val, median_val], yl, '--', 'Color', [0.3, 0.9, 0.3], ...
        'LineWidth', 4, 'DisplayName', 'Median');
    
    % Styling
    xlabel(ax_main, 'Number of Edges', 'FontSize', 32, 'FontWeight', 'bold');
    ylabel(ax_main, 'Probability Density', 'FontSize', 32, 'FontWeight', 'bold');
    title(ax_main, sprintf('PDF - %s', edge_type), 'FontSize', 36, 'FontWeight', 'bold');
    
    % Legend
    leg = legend(ax_main, 'Location', 'northeast', 'FontSize', 28);
    leg.Position(2) = leg.Position(2) - 0.15;
    
    grid(ax_main, 'on');
    set(ax_main, 'FontSize', 28, 'LineWidth', 2);
    ax_main.XAxis.LineWidth = 2.5;
    ax_main.YAxis.LineWidth = 2.5;
    
    % Add statistics text box
    text_str = sprintf('n=%d, μ=%.1f, σ=%.1f\nMedian=%.1f\nMin=%d, Max=%d', ...
        length(edge_counts), mean_val, std_val, median_val, min_val, max_val);
    text(ax_main, 0.98, 0.98, text_str, 'Units', 'normalized', ...
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
    [f, x] = ecdf(edge_counts);
    
    % Plot CDF
    plot(ax_cdf, x, f*100, 'LineWidth', 7, 'Color', color);
    
    % Calculate percentiles
    percentiles = [50, 90, 95, 99];
    p_vals = prctile(edge_counts, percentiles);
    
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
    xlabel(ax_cdf, 'Number of Edges', 'FontSize', 36, 'FontWeight', 'bold');
    ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 36, 'FontWeight', 'bold');
    title(ax_cdf, sprintf('CDF - %s', edge_type), 'FontSize', 38, 'FontWeight', 'bold');
    
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
        length(edge_counts), mean_val, std_val, median_val);
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