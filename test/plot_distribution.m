function plot_distribution(filter_name, frame_idx, output_dir)
% PLOT_DISTRIBUTION Plot PDF and CDF for filter distribution data
% Plots separate distributions for veridical (GT) and non-veridical edges
% Saves PDF and CDF as separate image files.

    if nargin < 2
        frame_idx = 0;
    end
    if nargin < 3
        output_dir = '../output_files';
    end
    
    % Construct filename
    filename = fullfile(output_dir, sprintf('%s_frame_%d.txt', filter_name, frame_idx));
    
    % Check if file exists
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    % Read the data (skip header lines starting with #)
    fid = fopen(filename, 'r');
    data = [];
    is_gt = [];
    header_done = false;
    
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(line)
            if line(1) == '#'
                continue;
            elseif ~header_done && contains(line, 'filter_value')
                header_done = true;
                continue;
            elseif header_done
                parts = strsplit(line, '\t');
                if length(parts) >= 2
                    value = str2double(parts{1});
                    gt_flag = str2double(parts{2});
                    if ~isnan(value) && ~isnan(gt_flag)
                        data = [data; value];
                        is_gt = [is_gt; gt_flag];
                    end
                end
            end
        end
    end
    fclose(fid);
    
    if isempty(data)
        error('No valid data found in file: %s', filename);
    end
    
    % Transform NCC scores to dissimilarity (1 - ncc_score)
    if strcmp(filter_name, 'ncc_score')
        data = 1 - data;
        filter_display_name = 'ncc dissimilarity (1 - ncc)';
    elseif strcmp(filter_name, 'location_error')
        filter_display_name = 'disparity difference';
    else
        filter_display_name = strrep(filter_name, '_', ' ');
    end
    
    % Separate GT and non-GT data
    data_gt = data(is_gt == 1);
    data_non_gt = data(is_gt == 0);
    
    fprintf('Loaded %d values from %s\n', length(data), filename);
    
    % Check which filter to determine if zoomed inset plot is needed
    is_ncc = strcmp(filter_name, 'ncc_score');
    is_epipolar = strcmp(filter_name, 'epipolar_distance');
    is_location = strcmp(filter_name, 'location_error');
    
    %% --- PLOT 1: PDF (With Zoomed Inset) ---
    % Create figure for PDF
    fig_pdf = figure('Position', [100, 100, 700, 500]);
    ax_main = axes('Parent', fig_pdf); % Explicitly create main axes
    hold(ax_main, 'on');
    
    % Define common bin edges
    data_min = min([data_gt; data_non_gt]);
    data_max = max([data_gt; data_non_gt]);
    bin_edges = linspace(data_min, data_max, 51);
    
    % Plot main histograms explicitly to ax_main
    histogram(ax_main, data_non_gt, bin_edges, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.2, 0.2], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Non-veridical');
    histogram(ax_main, data_gt, bin_edges, 'Normalization', 'pdf', 'FaceColor', [0.2, 0.6, 0.8], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Veridical');
    
    xlabel(ax_main, filter_display_name);
    ylabel(ax_main, 'Probability Density');
    title(ax_main, sprintf('PDF - %s (Frame %d)', filter_display_name, frame_idx));
    
    % Legend and Text
    leg = legend(ax_main, 'Location', 'northeast');
    leg.Position(2) = leg.Position(2) - 0.15;
    grid(ax_main, 'on');
    
    text_str = sprintf('Veridical: n=%d, μ=%.3f, σ=%.3f\nNon-veridical: n=%d, μ=%.3f, σ=%.3f', ...
                      length(data_gt), mean(data_gt), std(data_gt), ...
                      length(data_non_gt), mean(data_non_gt), std(data_non_gt));
    text(ax_main, 0.98, 0.98, text_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
         'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black', 'Interpreter', 'none');
    
    hold(ax_main, 'off');
    
    % --- INSET LOGIC ---
    % Calculate position based on the main axes
    main_pos = get(ax_main, 'Position'); % [left, bottom, width, height]
    
    inset_width = main_pos(3) * 0.38;
    inset_height = main_pos(4) * 0.38;
    inset_left = main_pos(1) + main_pos(3) * 0.32;
    inset_bottom = main_pos(2) + main_pos(4) * 0.42;
    
    % Initialize variables for zoom logic
    plot_inset = false;
    zoom_min = 0; zoom_max = 0;
    zoom_title = '';
    
    if is_ncc
        plot_inset = true;
        zoom_min = 0; zoom_max = 0.5;
        zoom_title = 'Zoomed: NCC 0.5-1.0';
    elseif is_epipolar
        plot_inset = true;
        zoom_min = 0; zoom_max = 2;
        zoom_title = 'Zoomed: EP 1-3 pixels';
    elseif is_location
        plot_inset = true;
        zoom_min = 0; zoom_max = 20;
        zoom_title = 'Zoomed: LP 0-20 pixels';
    end
    
    if plot_inset
        % Create inset axes explicitly on the PDF figure
        ax_inset = axes('Parent', fig_pdf, 'Position', [inset_left, inset_bottom, inset_width, inset_height]);
        hold(ax_inset, 'on');
        
        % Filter data
        data_gt_zoom = data_gt(data_gt >= zoom_min & data_gt <= zoom_max);
        data_non_gt_zoom = data_non_gt(data_non_gt >= zoom_min & data_non_gt <= zoom_max);
        bin_edges_zoom = linspace(zoom_min, zoom_max, 31);
        
        % Plot Histograms explicitly to ax_inset
        histogram(ax_inset, data_non_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.2, 0.2], ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.6);
        histogram(ax_inset, data_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', 'FaceColor', [0.2, 0.6, 0.8], ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.6);
        
        % Styling
        xlabel(ax_inset, filter_display_name, 'FontSize', 8);
        ylabel(ax_inset, 'PDF', 'FontSize', 8);
        title(ax_inset, zoom_title, 'FontSize', 9);
        grid(ax_inset, 'on');
        xlim(ax_inset, [zoom_min, zoom_max]);
        set(ax_inset, 'FontSize', 8);
        box(ax_inset, 'on');
        hold(ax_inset, 'off');
        
        % Arrow Annotation (Explicitly attach to fig_pdf)
        arrow_x = [inset_left + inset_width*0.1, main_pos(1) + main_pos(3)*0.05];
        arrow_y = [inset_bottom, main_pos(2) + main_pos(4)*0.25];
        annotation(fig_pdf, 'arrow', arrow_x, arrow_y, 'LineWidth', 1.5, 'Color', 'k');
    end
    
    % Save PDF Figure
    save_filename_pdf = fullfile(output_dir, sprintf('%s_frame_%d_pdf.png', filter_name, frame_idx));
    saveas(fig_pdf, save_filename_pdf);
    fprintf('Saved PDF figure to: %s\n', save_filename_pdf);
    close(fig_pdf);
    
    
    %% --- PLOT 2: CDF ---
    fig_cdf = figure('Position', [150, 150, 700, 500]);
    ax_cdf = axes('Parent', fig_cdf);
    hold(ax_cdf, 'on');
    
    [f_non_gt, x_non_gt] = ecdf(data_non_gt);
    [f_gt, x_gt] = ecdf(data_gt);
    plot(ax_cdf, x_non_gt, f_non_gt*100, 'LineWidth', 2.5, 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'Non-veridical');
    plot(ax_cdf, x_gt, f_gt*100, 'LineWidth', 2.5, 'Color', [0.2, 0.6, 0.8], 'DisplayName', 'Veridical');
    
    xlabel(ax_cdf, filter_display_name, 'FontSize', 12);
    ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 12);
    title(ax_cdf, sprintf('CDF - %s (Frame %d)', filter_display_name, frame_idx), 'FontSize', 14);
    legend(ax_cdf, 'Location', 'northwest', 'FontSize', 11);
    grid(ax_cdf, 'on');
    
    % Add percentile markers for both distributions
    percentiles = [50, 90, 95, 99];
    percentile_values_gt = prctile(data_gt, percentiles);
    percentile_values_non_gt = prctile(data_non_gt, percentiles);
    
    x_lim = xlim(ax_cdf);
    x_range = x_lim(2) - x_lim(1);
    
    % Plot percentile lines for veridical (blue)
    for i = 1:length(percentiles)
        plot(ax_cdf, [percentile_values_gt(i), percentile_values_gt(i)], [0, percentiles(i)], ...
             ':', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(ax_cdf, percentile_values_gt(i), percentiles(i), 'o', ...
             'MarkerSize', 7, 'MarkerFaceColor', [0.2, 0.6, 0.8], 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');
        text(ax_cdf, percentile_values_gt(i) + x_range*0.01, percentiles(i) + 2, ...
             sprintf('p%d:%.2f', percentiles(i), percentile_values_gt(i)), ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
             'FontSize', 8, 'Color', [0.2, 0.6, 0.8], 'FontWeight', 'bold', ...
             'BackgroundColor', [1 1 1 0.7]);
    end
    
    % Plot percentile lines for non-veridical (red)
    for i = 1:length(percentiles)
        plot(ax_cdf, [percentile_values_non_gt(i), percentile_values_non_gt(i)], [0, percentiles(i)], ...
             ':', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(ax_cdf, percentile_values_non_gt(i), percentiles(i), 's', ...
             'MarkerSize', 7, 'MarkerFaceColor', [0.8, 0.2, 0.2], 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');
        text(ax_cdf, percentile_values_non_gt(i) + x_range*0.01, percentiles(i) - 2, ...
             sprintf('p%d:%.2f', percentiles(i), percentile_values_non_gt(i)), ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
             'FontSize', 8, 'Color', [0.8, 0.2, 0.2], 'FontWeight', 'bold', ...
             'BackgroundColor', [1 1 1 0.7]);
    end
    
    % Add threshold lines for common distance thresholds (useful for distance distributions)
    if contains(filter_name, 'Distance') || contains(filter_name, 'distance') || contains(filter_name, 'GT')
        thresholds = [0.5, 1.0, 2.0, 3.0, 5.0];
        x_lim = xlim(ax_cdf);
        
        % Ensure unique x values for interpolation
        [x_gt_unique, ia_gt] = unique(x_gt);
        f_gt_unique = f_gt(ia_gt);
        [x_non_gt_unique, ia_non_gt] = unique(x_non_gt);
        f_non_gt_unique = f_non_gt(ia_non_gt);
        
        for threshold = thresholds
            if threshold <= x_lim(2)
                % Find percentages at this threshold for both distributions
                if length(x_gt_unique) > 1
                    pct_gt = interp1(x_gt_unique, f_gt_unique*100, threshold, 'linear', 'extrap');
                else
                    pct_gt = f_gt_unique(1)*100;
                end
                
                if length(x_non_gt_unique) > 1
                    pct_non_gt = interp1(x_non_gt_unique, f_non_gt_unique*100, threshold, 'linear', 'extrap');
                else
                    pct_non_gt = f_non_gt_unique(1)*100;
                end
                
                % Draw vertical line
                plot(ax_cdf, [threshold, threshold], [0, 100], 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
                
                % Annotate with percentages
                text(ax_cdf, threshold, 5, sprintf('%.0fpx', threshold), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                     'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold', ...
                     'BackgroundColor', [1 1 1 0.7]);
                
                % Mark points on curves
                plot(ax_cdf, threshold, pct_gt, 'o', 'MarkerSize', 6, ...
                     'MarkerFaceColor', [0.2, 0.6, 0.8], 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
                     'HandleVisibility', 'off');
                plot(ax_cdf, threshold, pct_non_gt, 'o', 'MarkerSize', 6, ...
                     'MarkerFaceColor', [0.8, 0.2, 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
                     'HandleVisibility', 'off');
            end
        end
    end
    
    % Add summary statistics box (moved to right side)
    stats_str = sprintf('Veridical:\n  n=%d\n  p50=%.2f\n  p90=%.2f\n  p95=%.2f\n  p99=%.2f\n\nNon-veridical:\n  n=%d\n  p50=%.2f\n  p90=%.2f\n  p95=%.2f\n  p99=%.2f', ...
                        length(data_gt), percentile_values_gt(1), percentile_values_gt(2), percentile_values_gt(3), percentile_values_gt(4), ...
                        length(data_non_gt), percentile_values_non_gt(1), percentile_values_non_gt(2), percentile_values_non_gt(3), percentile_values_non_gt(4));
    text(ax_cdf, 0.98, 0.55, stats_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
         'FontSize', 9, 'BackgroundColor', [1 1 1 0.9], 'EdgeColor', 'black', ...
         'Interpreter', 'none', 'FontName', 'FixedWidth');
    
    ylim(ax_cdf, [0, 100]);
    
    hold(ax_cdf, 'off');
    
    % Save CDF Figure
    save_filename_cdf = fullfile(output_dir, sprintf('%s_frame_%d_cdf.png', filter_name, frame_idx));
    saveas(fig_cdf, save_filename_cdf);
    fprintf('Saved CDF figure to: %s\n', save_filename_cdf);
    close(fig_cdf);
end