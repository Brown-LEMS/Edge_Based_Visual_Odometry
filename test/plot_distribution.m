% filepath: test/plot_distribution.m
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
    
    % Construct filename with NEW subdirectory structure
    % output_files/values/{filter_name}/frame_{idx}.txt
    filename = fullfile(output_dir, 'values', filter_name, sprintf('frame_%d.txt', frame_idx));
    
    % Check if file exists
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    % Read the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    % Skip header lines (lines starting with #)
    line = fgetl(fid);
    while ischar(line) && startsWith(line, '#')
        line = fgetl(fid);
    end
    
    % Read data (filter_value, is_veridical)
    data = textscan(fid, '%f %d', 'HeaderLines', 0);
    fclose(fid);
    
    if isempty(data{1})
        error('No data found in file: %s', filename);
    end
    
    filter_values = data{1};
    is_veridical = data{2};
    
    % Transform NCC scores to dissimilarity
    is_ncc_filter = contains(filter_name, 'ncc');
    if is_ncc_filter
        filter_values = 1 - filter_values;
        if contains(filter_name, 'temporal')
            side = regexp(filter_name, '(left|right)', 'match', 'once');
            filter_display_name = sprintf('Temporal 1-NCC (%s)', upper(side));
        else
            filter_display_name = '1 - NCC';
        end
    elseif strcmp(filter_name, 'location')
        filter_display_name = 'Disparity';
    elseif contains(filter_name, 'temporal_location_error')
        side = regexp(filter_name, '(left|right)', 'match', 'once');
        filter_display_name = sprintf('Disparity (%s)', upper(side));
    elseif strcmp(filter_name, 'sift')
        filter_display_name = 'SIFT';
    elseif contains(filter_name, 'temporal_sift_distance')
        side = regexp(filter_name, '(left|right)', 'match', 'once');
        filter_display_name = sprintf('Temporal SIFT (%s)', upper(side));
    elseif strcmp(filter_name, 'epipolar')
        filter_display_name = 'Epipolar Proximity';
    elseif contains(filter_name, 'temporal_orientation_difference')
        side = regexp(filter_name, '(left|right)', 'match', 'once');
        filter_display_name = sprintf('Temporal Orientation Difference (degrees) (%s)', upper(side));
    elseif strcmp(filter_name, 'orientation')
        filter_display_name = 'Orientation Difference (degrees)';
    else
        filter_display_name = strrep(filter_name, '_', ' ');
    end
    
    % Separate GT and non-GT
    data_gt = filter_values(is_veridical == 1);
    data_non_gt = filter_values(is_veridical == 0);
    
    fprintf('Loaded %d samples from %s\n', length(filter_values), filename);
    fprintf('  GT samples: %d\n', length(data_gt));
    fprintf('  Non-GT samples: %d\n', length(data_non_gt));
    
    % Define threshold values for temporal and stereo filters
    threshold_value = [];
    threshold_label = '';
    % Temporal filters
    if contains(filter_name, 'temporal_location_error')
        threshold_value = 30;
        threshold_label = 'Threshold = 30 px';
    elseif contains(filter_name, 'temporal_orientation_difference')
        threshold_value = 10;
        threshold_label = 'Threshold = 10°';
    elseif contains(filter_name, 'temporal_ncc')
        threshold_value = 0.2;  % 1 - 0.8 = 0.2 (dissimilarity)
        threshold_label = 'Threshold = 0.2';
    elseif contains(filter_name, 'temporal_sift_distance')
        threshold_value = 200;
        threshold_label = 'Threshold = 200';
    % Stereo filters
    elseif strcmp(filter_name, 'epipolar')
        threshold_value = 0.5;
        threshold_label = 'Threshold = 0.5 px';
    elseif strcmp(filter_name, 'location')
        threshold_value = 25;
        threshold_label = 'Threshold = 25 px';
    elseif strcmp(filter_name, 'orientation')
        threshold_value = 10.0;
        threshold_label = 'Threshold = 10°';
    elseif strcmp(filter_name, 'sift')
        threshold_value = 500;
        threshold_label = 'Threshold = 500';
    elseif strcmp(filter_name, 'ncc')
        threshold_value = 0.4;  % 1 - 0.6 = 0.4 (dissimilarity)
        threshold_label = 'Threshold = 0.4';
    end
    
    % Check which filter for zoomed inset
    is_epipolar = strcmp(filter_name, 'epipolar');
    is_location = strcmp(filter_name, 'location');
    is_temporal_location = contains(filter_name, 'temporal_location_error');
    is_sift = strcmp(filter_name, 'sift');
    is_temporal_sift = contains(filter_name, 'temporal_sift_distance');
    is_orientation = strcmp(filter_name, 'orientation');
    is_temporal_orientation = contains(filter_name, 'temporal_orientation_difference');
    
    %% --- PLOT 1: PDF ---
    fig_pdf = figure('Position', [100, 100, 1200, 900]);
    ax_main = axes('Parent', fig_pdf);
    hold(ax_main, 'on');
    
    % Define bin edges
    data_min = min([data_gt; data_non_gt]);
    data_max = max([data_gt; data_non_gt]);
    bin_edges = linspace(data_min, data_max, 51);
    
    % Plot histograms
    histogram(ax_main, data_non_gt, bin_edges, 'Normalization', 'pdf', ...
        'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
        'DisplayName', 'Non-veridical');
    histogram(ax_main, data_gt, bin_edges, 'Normalization', 'pdf', ...
        'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
        'DisplayName', 'Veridical');
    
    % Add threshold vertical line if defined
    if ~isempty(threshold_value)
        yl = ylim(ax_main);
        plot(ax_main, [threshold_value, threshold_value], yl, 'g--', 'LineWidth', 4, ...
            'HandleVisibility', 'off');
        xl = xlim(ax_main);
        x_offset = (xl(2) - xl(1)) * 0.02;
        % Special positioning for NCC - higher up at 85%
        if is_ncc_filter
            text(ax_main, threshold_value + x_offset, yl(2)*0.85, threshold_label, ...
                'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5);
        % Special positioning for SIFT - to the right at middle height
        elseif is_temporal_sift || is_sift
            text(ax_main, threshold_value + x_offset, yl(2)*0.5, threshold_label, ...
                'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5);
        else
            % Default positioning - centered at top
            text(ax_main, threshold_value, yl(2)*0.9, threshold_label, ...
                'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5);
        end
    end
    
    % Styling
    xlabel(ax_main, filter_display_name, 'FontSize', 32, 'FontWeight', 'bold');
    ylabel(ax_main, 'Probability Density', 'FontSize', 32, 'FontWeight', 'bold');
    
    % Legend
    leg = legend(ax_main, 'Location', 'northeast', 'FontSize', 28);
    leg.Position(2) = leg.Position(2) - 0.15;
    
    grid(ax_main, 'on');
    set(ax_main, 'FontSize', 28, 'LineWidth', 2);
    ax_main.XAxis.LineWidth = 2.5;
    ax_main.YAxis.LineWidth = 2.5;
    
    % Statistics text
    text_str = sprintf('Veridical: n=%d, μ=%.3f, σ=%.3f\nNon-veridical: n=%d, μ=%.3f, σ=%.3f', ...
        length(data_gt), mean(data_gt), std(data_gt), ...
        length(data_non_gt), mean(data_non_gt), std(data_non_gt));
    text(ax_main, 0.98, 0.98, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
        'FontSize', 24, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'Interpreter', 'none', 'LineWidth', 2);
    
    hold(ax_main, 'off');
    
    % --- INSET LOGIC ---
    main_pos = get(ax_main, 'Position');
    inset_width = main_pos(3) * 0.38;
    inset_height = main_pos(4) * 0.35;
    inset_left = main_pos(1) + main_pos(3) * 0.32;
    inset_bottom = main_pos(2) + main_pos(4) * 0.15;
    
    plot_inset = false;
    zoom_min = 0; 
    zoom_max = 0;
    zoom_title = '';
    
    if is_ncc_filter
        plot_inset = true;
        zoom_min = 0; 
        zoom_max = 0.5;
        zoom_title = 'Zoomed: 1-NCC [0, 0.5]';
    elseif is_epipolar
        plot_inset = true;
        zoom_min = 0; 
        zoom_max = 2;
        zoom_title = 'Zoomed: EP 0-2 pixels';
    elseif is_location || is_temporal_location
        plot_inset = true;
        zoom_min = 0; 
        zoom_max = 30;
        if is_temporal_location
            zoom_title = 'Zoomed: Temporal LD 0-30 pixels';
        else
            zoom_title = 'Zoomed: LP 0-20 pixels';
        end
    elseif is_orientation || is_temporal_orientation
        plot_inset = true;
        zoom_min = 0; 
        zoom_max = 40;
        if is_temporal_orientation
            zoom_title = 'Zoomed: Temporal OD 0-40 degrees';
        else
            zoom_title = 'Zoomed: OD 0-40 degrees';
        end
    end
    
    if plot_inset
        % Create inset axes
        ax_inset = axes('Parent', fig_pdf, 'Position', [inset_left, inset_bottom, inset_width, inset_height]);
        hold(ax_inset, 'on');
        
        % Filter data
        data_gt_zoom = data_gt(data_gt >= zoom_min & data_gt <= zoom_max);
        data_non_gt_zoom = data_non_gt(data_non_gt >= zoom_min & data_non_gt <= zoom_max);
        bin_edges_zoom = linspace(zoom_min, zoom_max, 31);
        
        % Plot histograms
        histogram(ax_inset, data_non_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', ...
            'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        histogram(ax_inset, data_gt_zoom, bin_edges_zoom, 'Normalization', 'pdf', ...
            'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        
        % Add threshold line to inset if it falls within zoom range
        if ~isempty(threshold_value) && threshold_value >= zoom_min && threshold_value <= zoom_max
            yl_inset = ylim(ax_inset);
            plot(ax_inset, [threshold_value, threshold_value], yl_inset, 'g--', 'LineWidth', 3);
        end
        
        % Styling
        xlabel(ax_inset, filter_display_name, 'FontSize', 22, 'FontWeight', 'bold');
        ylabel(ax_inset, 'PDF', 'FontSize', 22, 'FontWeight', 'bold');
        title(ax_inset, zoom_title, 'FontSize', 24, 'FontWeight', 'bold');
        grid(ax_inset, 'on');
        xlim(ax_inset, [zoom_min, zoom_max]);
        set(ax_inset, 'FontSize', 20, 'LineWidth', 2);
        ax_inset.XAxis.LineWidth = 2;
        ax_inset.YAxis.LineWidth = 2;
        ax_inset.Color = [1 1 1];
        ax_inset.Box = 'on';
        ax_inset.LineWidth = 3;
        
        hold(ax_inset, 'off');
        
        % Arrow pointing to zoomed region
        ax_main.Units = 'normalized';
        xl = xlim(ax_main);
        target_data_x = (zoom_min + zoom_max) / 2;
        x_ratio = (target_data_x - xl(1)) / (xl(2) - xl(1));
        arrow_end_x = main_pos(1) + (x_ratio * main_pos(3));
        arrow_end_y = main_pos(2);
        arrow_start_x = inset_left;
        arrow_start_y = inset_bottom;
        
        annotation(fig_pdf, 'arrow', [arrow_start_x, arrow_end_x], [arrow_start_y, arrow_end_y], ...
            'LineWidth', 3.5, 'Color', [0.2, 0.2, 0.2], ...
            'HeadStyle', 'cback1', 'HeadLength', 15, 'HeadWidth', 15);
    end
    
    % Save PDF figure
    save_filename_pdf = fullfile(output_dir, 'values', filter_name, sprintf('frame_%d_pdf.png', frame_idx));
    saveas(fig_pdf, save_filename_pdf);
    fprintf('Saved PDF figure to: %s\n', save_filename_pdf);
    close(fig_pdf);
    
  %% --- PLOT 2: CDF (Final Publication Quality) ---
    fig_cdf = figure('Position', [150, 150, 1200, 900]);
    ax_cdf = axes('Parent', fig_cdf);
    hold(ax_cdf, 'on');
    
    [f_non_gt, x_non_gt] = ecdf(data_non_gt);
    [f_gt, x_gt] = ecdf(data_gt);
    
    plot(ax_cdf, x_non_gt, f_non_gt*100, 'LineWidth', 7, 'Color', [0.8, 0.2, 0.2], 'DisplayName', 'Non-veridical');
    plot(ax_cdf, x_gt, f_gt*100, 'LineWidth', 7, 'Color', [0.2, 0.6, 0.8], 'DisplayName', 'Veridical');
    
    % Add threshold vertical line if defined
    if ~isempty(threshold_value)
        plot(ax_cdf, [threshold_value, threshold_value], [0, 100], 'g--', 'LineWidth', 4, ...
            'HandleVisibility', 'off');
        xl_cdf_range = xlim(ax_cdf);
        x_offset = (xl_cdf_range(2) - xl_cdf_range(1)) * 0.02;
        % Special positioning for SIFT - lower at 25% (below legend)
        if is_sift || is_temporal_sift
            text(ax_cdf, threshold_value + x_offset, 25, threshold_label, ...
                'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
        % Special positioning for location, NCC, orientation, epipolar - to the right at 70%
        elseif is_temporal_location || is_ncc_filter || is_temporal_orientation || is_orientation || is_epipolar || is_location
            text(ax_cdf, threshold_value + x_offset, 70, threshold_label, ...
                'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
        else
            % Default positioning for other filters - centered at top
            text(ax_cdf, threshold_value, 105, threshold_label, ...
                'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
        end
    end
    
    percentiles = [50, 90, 95, 99];
    p_gt = prctile(data_gt, percentiles);
    p_non_gt = prctile(data_non_gt, percentiles);
    
    xlabel(ax_cdf, filter_display_name, 'FontSize', 36, 'FontWeight', 'bold');
    ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 36, 'FontWeight', 'bold');
    
    grid(ax_cdf, 'on');
    set(ax_cdf, 'FontSize', 32, 'LineWidth', 3);
    xl_cdf = xlim(ax_cdf); xr_cdf = xl_cdf(2) - xl_cdf(1);

    % Percentile Annotations (All Right-Aligned and Large)
    for i = 1:length(percentiles)
        plot(ax_cdf, [p_gt(i), p_gt(i)], [0, percentiles(i)], ':', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 3, 'HandleVisibility', 'off');
        text(ax_cdf, p_gt(i) + xr_cdf*0.02, percentiles(i) + 1.5, sprintf('p%d: %.1f', percentiles(i), p_gt(i)), ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'FontSize', 28, 'Color', [0.2, 0.6, 0.8], 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.4]);
        
        plot(ax_cdf, [p_non_gt(i), p_non_gt(i)], [0, percentiles(i)], ':', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 3, 'HandleVisibility', 'off');
        text(ax_cdf, p_non_gt(i) + xr_cdf*0.02, percentiles(i) - 1.5, sprintf('p%d: %.1f', percentiles(i), p_non_gt(i)), ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 28, 'Color', [0.8, 0.2, 0.2], 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.4]);
    end

    % Threshold Markers
    thresholds = [0.5, 1.0, 2.0];
    for t = thresholds
        if t <= xl_cdf(2)
            plot(ax_cdf, [t, t], [0, 100], 'k--', 'LineWidth', 2.5, 'HandleVisibility', 'off');
            text(ax_cdf, t, 101.5, sprintf('%.1f', t), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 28, 'FontWeight', 'bold');
        end
    end
    
    % Compact Stats Box (Southeast)
    stats_str = sprintf('Veridical: p50=%.1f, p99=%.1f (n=%d)\nNon-veridical: p50=%.1f, p99=%.1f (n=%d)', ...
                        p_gt(1), p_gt(4), length(data_gt), p_non_gt(1), p_non_gt(4), length(data_non_gt));
    text(ax_cdf, 0.98, 0.08, stats_str, 'Units', 'normalized', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
         'FontSize', 26, 'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'black', 'LineWidth', 2, 'Margin', 8);

    % CUSTOM LOWERED LEGEND
    leg_cdf = legend(ax_cdf, 'FontSize', 32);
    set(leg_cdf, 'Units', 'normalized', 'Position', [0.65, 0.60, 0.25, 0.1]); 
    
    ylim(ax_cdf, [0, 108]); 
    hold(ax_cdf, 'off');
    
    % Save CDF figure
    save_filename_cdf = fullfile(output_dir, 'values', filter_name, sprintf('frame_%d_cdf.png', frame_idx));
    saveas(fig_cdf, save_filename_cdf);
    fprintf('Saved CDF figure to: %s\n', save_filename_cdf);
    close(fig_cdf);
end