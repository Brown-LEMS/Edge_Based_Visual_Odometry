% filepath: test/plot_distribution_left_right.m
function plot_distribution_left_right(filter_base_name, frame_idx, output_dir)
    % PLOT_DISTRIBUTION_LEFT_RIGHT Plot combined left and right distributions
    % Plots left and right temporal filter distributions together
    % Colors: Left veridical (blue), Left non-veridical (red)
    %         Right veridical (purple), Right non-veridical (yellow)
    
    if nargin < 2
        frame_idx = 0;
    end
    if nargin < 3
        output_dir = '../outputs';
    end
    
    % Construct filenames for left and right
    filter_name_left = [filter_base_name '_left'];
    filter_name_right = [filter_base_name '_right'];
    
    filename_left = fullfile(output_dir, 'values', filter_name_left, sprintf('frame_%d.txt', frame_idx));
    filename_right = fullfile(output_dir, 'values', filter_name_right, sprintf('frame_%d.txt', frame_idx));
    
    % Check if files exist
    if ~exist(filename_left, 'file')
        error('Left file not found: %s', filename_left);
    end
    if ~exist(filename_right, 'file')
        error('Right file not found: %s', filename_right);
    end
    
    % Read left data
    fid_left = fopen(filename_left, 'r');
    if fid_left == -1
        error('Could not open left file: %s', filename_left);
    end
    line = fgetl(fid_left);
    while ischar(line) && startsWith(line, '#')
        line = fgetl(fid_left);
    end
    data_left = textscan(fid_left, '%f %d', 'HeaderLines', 0);
    fclose(fid_left);
    
    % Read right data
    fid_right = fopen(filename_right, 'r');
    if fid_right == -1
        error('Could not open right file: %s', filename_right);
    end
    line = fgetl(fid_right);
    while ischar(line) && startsWith(line, '#')
        line = fgetl(fid_right);
    end
    data_right = textscan(fid_right, '%f %d', 'HeaderLines', 0);
    fclose(fid_right);
    
    if isempty(data_left{1}) || isempty(data_right{1})
        error('No data found in one or both files');
    end
    
    filter_values_left = data_left{1};
    is_veridical_left = data_left{2};
    filter_values_right = data_right{1};
    is_veridical_right = data_right{2};
    
    % Transform NCC scores to dissimilarity if needed
    is_ncc_filter = contains(filter_base_name, 'ncc');
    if is_ncc_filter
        filter_values_left = 1 - filter_values_left;
        filter_values_right = 1 - filter_values_right;
        filter_display_name = 'Temporal 1-NCC';
    elseif contains(filter_base_name, 'location_error')
        filter_display_name = 'Temporal Location Error';
    elseif contains(filter_base_name, 'sift_distance')
        filter_display_name = 'Temporal SIFT Distance';
    elseif contains(filter_base_name, 'orientation_difference')
        filter_display_name = 'Temporal Orientation Difference';
    else
        filter_display_name = strrep(filter_base_name, '_', ' ');
    end
    
    % Separate by veridical status
    left_gt = filter_values_left(is_veridical_left == 1);
    left_non_gt = filter_values_left(is_veridical_left == 0);
    right_gt = filter_values_right(is_veridical_right == 1);
    right_non_gt = filter_values_right(is_veridical_right == 0);
    
    fprintf('Loaded data for %s\n', filter_display_name);
    fprintf('  Left - GT: %d, Non-GT: %d\n', length(left_gt), length(left_non_gt));
    fprintf('  Right - GT: %d, Non-GT: %d\n', length(right_gt), length(right_non_gt));
    
    % Define colors
    color_left_veridical = [0.2, 0.6, 0.8];      % Blue
    color_left_non_veridical = [0.8, 0.2, 0.2];  % Red
    color_right_veridical = [0.6, 0.2, 0.8];     % Purple
    color_right_non_veridical = [0.9, 0.8, 0.0]; % Yellow
    
    % Define threshold values for temporal filters
    threshold_value = [];
    threshold_label = '';
    if contains(filter_base_name, 'location_error')
        threshold_value = 30;
        threshold_label = 'Threshold = 30 px';
    elseif contains(filter_base_name, 'orientation_difference')
        threshold_value = 10;
        threshold_label = 'Threshold = 10°';
    elseif contains(filter_base_name, 'ncc')
        threshold_value = 0.2;  % 1 - 0.8 = 0.2 (dissimilarity)
        threshold_label = 'Threshold = 0.2';
    elseif contains(filter_base_name, 'sift_distance')
        threshold_value = 200;
        threshold_label = 'Threshold = 200';
    end
    
    % Check if we need zoomed inset
    is_location = contains(filter_base_name, 'location_error');
    is_orientation = contains(filter_base_name, 'orientation_difference');
    
    %% --- PLOT 1: PDF ---
    fig_pdf = figure('Position', [100, 100, 1200, 900]);
    ax_main = axes('Parent', fig_pdf);
    hold(ax_main, 'on');
    
    % Compute KDE for all distributions
    data_min = min([left_gt; left_non_gt; right_gt; right_non_gt]);
    data_max = max([left_gt; left_non_gt; right_gt; right_non_gt]);
    x_range = linspace(data_min, data_max, 200);
    
    % Plot KDE curves
    if ~isempty(left_non_gt)
        [f_left_non_gt, xi_left_non_gt] = ksdensity(left_non_gt, x_range);
        plot(ax_main, xi_left_non_gt, f_left_non_gt, 'LineWidth', 5, 'Color', color_left_non_veridical, ...
            'DisplayName', 'Left Non-veridical');
    end
    if ~isempty(left_gt)
        [f_left_gt, xi_left_gt] = ksdensity(left_gt, x_range);
        plot(ax_main, xi_left_gt, f_left_gt, 'LineWidth', 5, 'Color', color_left_veridical, ...
            'DisplayName', 'Left Veridical');
    end
    if ~isempty(right_non_gt)
        [f_right_non_gt, xi_right_non_gt] = ksdensity(right_non_gt, x_range);
        plot(ax_main, xi_right_non_gt, f_right_non_gt, 'LineWidth', 5, 'Color', color_right_non_veridical, ...
            'LineStyle', '--', 'DisplayName', 'Right Non-veridical');
    end
    if ~isempty(right_gt)
        [f_right_gt, xi_right_gt] = ksdensity(right_gt, x_range);
        plot(ax_main, xi_right_gt, f_right_gt, 'LineWidth', 5, 'Color', color_right_veridical, ...
            'LineStyle', '--', 'DisplayName', 'Right Veridical');
    end
    
    % Add threshold vertical line if defined
    if ~isempty(threshold_value)
        yl = ylim(ax_main);
        plot(ax_main, [threshold_value, threshold_value], yl, 'g--', 'LineWidth', 4, ...
            'DisplayName', threshold_label);
    end
    
    % Styling
    xlabel(ax_main, filter_display_name, 'FontSize', 32, 'FontWeight', 'bold');
    ylabel(ax_main, 'Probability Density', 'FontSize', 32, 'FontWeight', 'bold');
    title(ax_main, sprintf('PDF - %s (Left & Right)', filter_display_name), ...
        'FontSize', 36, 'FontWeight', 'bold');
    
    % Legend
    leg = legend(ax_main, 'Location', 'northeast', 'FontSize', 24);
    leg.Position(2) = leg.Position(2) - 0.15;
    
    grid(ax_main, 'on');
    set(ax_main, 'FontSize', 28, 'LineWidth', 2);
    ax_main.XAxis.LineWidth = 2.5;
    ax_main.YAxis.LineWidth = 2.5;
    
    % Statistics text
    text_str = sprintf(['Left: Veridical n=%d, μ=%.2f | Non-veridical n=%d, μ=%.2f\n' ...
                       'Right: Veridical n=%d, μ=%.2f | Non-veridical n=%d, μ=%.2f'], ...
        length(left_gt), mean(left_gt), length(left_non_gt), mean(left_non_gt), ...
        length(right_gt), mean(right_gt), length(right_non_gt), mean(right_non_gt));
    text(ax_main, 0.98, 0.98, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
        'FontSize', 20, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
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
    elseif is_location
        plot_inset = true;
        zoom_min = 0;
        zoom_max = 30;
        zoom_title = 'Zoomed: Temporal LE 0-30 pixels';
    elseif is_orientation
        plot_inset = true;
        zoom_min = 0;
        zoom_max = 40;
        zoom_title = 'Zoomed: Temporal OD 0-40 degrees';
    end
    
    if plot_inset
        % Create inset axes
        ax_inset = axes('Parent', fig_pdf, 'Position', [inset_left, inset_bottom, inset_width, inset_height]);
        hold(ax_inset, 'on');
        
        % Filter data
        left_gt_zoom = left_gt(left_gt >= zoom_min & left_gt <= zoom_max);
        left_non_gt_zoom = left_non_gt(left_non_gt >= zoom_min & left_non_gt <= zoom_max);
        right_gt_zoom = right_gt(right_gt >= zoom_min & right_gt <= zoom_max);
        right_non_gt_zoom = right_non_gt(right_non_gt >= zoom_min & right_non_gt <= zoom_max);
        x_range_zoom = linspace(zoom_min, zoom_max, 150);
        
        % Plot KDE curves for zoomed region
        if ~isempty(left_non_gt_zoom)
            [f_left_non_gt_zoom, xi_left_non_gt_zoom] = ksdensity(left_non_gt_zoom, x_range_zoom);
            plot(ax_inset, xi_left_non_gt_zoom, f_left_non_gt_zoom, 'LineWidth', 4, 'Color', color_left_non_veridical);
        end
        if ~isempty(left_gt_zoom)
            [f_left_gt_zoom, xi_left_gt_zoom] = ksdensity(left_gt_zoom, x_range_zoom);
            plot(ax_inset, xi_left_gt_zoom, f_left_gt_zoom, 'LineWidth', 4, 'Color', color_left_veridical);
        end
        if ~isempty(right_non_gt_zoom)
            [f_right_non_gt_zoom, xi_right_non_gt_zoom] = ksdensity(right_non_gt_zoom, x_range_zoom);
            plot(ax_inset, xi_right_non_gt_zoom, f_right_non_gt_zoom, 'LineWidth', 4, 'Color', color_right_non_veridical, 'LineStyle', '--');
        end
        if ~isempty(right_gt_zoom)
            [f_right_gt_zoom, xi_right_gt_zoom] = ksdensity(right_gt_zoom, x_range_zoom);
            plot(ax_inset, xi_right_gt_zoom, f_right_gt_zoom, 'LineWidth', 4, 'Color', color_right_veridical, 'LineStyle', '--');
        end
        
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
    save_filename_pdf = fullfile(output_dir, 'values', filter_base_name, sprintf('frame_%d_combined_pdf.png', frame_idx));
    % Create directory if it doesn't exist
    if ~exist(fullfile(output_dir, 'values', filter_base_name), 'dir')
        mkdir(fullfile(output_dir, 'values', filter_base_name));
    end
    saveas(fig_pdf, save_filename_pdf);
    fprintf('Saved combined PDF figure to: %s\n', save_filename_pdf);
    close(fig_pdf);
    
    %% --- PLOT 2: CDF ---
    fig_cdf = figure('Position', [150, 150, 1200, 900]);
    ax_cdf = axes('Parent', fig_cdf);
    hold(ax_cdf, 'on');
    
    [f_left_non_gt, x_left_non_gt] = ecdf(left_non_gt);
    [f_left_gt, x_left_gt] = ecdf(left_gt);
    [f_right_non_gt, x_right_non_gt] = ecdf(right_non_gt);
    [f_right_gt, x_right_gt] = ecdf(right_gt);
    
    plot(ax_cdf, x_left_non_gt, f_left_non_gt*100, 'LineWidth', 6, 'Color', color_left_non_veridical, ...
        'DisplayName', 'Left Non-veridical');
    plot(ax_cdf, x_left_gt, f_left_gt*100, 'LineWidth', 6, 'Color', color_left_veridical, ...
        'DisplayName', 'Left Veridical');
    plot(ax_cdf, x_right_non_gt, f_right_non_gt*100, 'LineWidth', 6, 'Color', color_right_non_veridical, ...
        'DisplayName', 'Right Non-veridical', 'LineStyle', '--');
    plot(ax_cdf, x_right_gt, f_right_gt*100, 'LineWidth', 6, 'Color', color_right_veridical, ...
        'DisplayName', 'Right Veridical', 'LineStyle', '--');
    
    % Add threshold vertical line if defined
    if ~isempty(threshold_value)
        plot(ax_cdf, [threshold_value, threshold_value], [0, 100], 'g--', 'LineWidth', 4, ...
            'DisplayName', threshold_label);
    end
    
    percentiles = [50, 90, 95, 99];
    p_left_gt = prctile(left_gt, percentiles);
    p_left_non_gt = prctile(left_non_gt, percentiles);
    p_right_gt = prctile(right_gt, percentiles);
    p_right_non_gt = prctile(right_non_gt, percentiles);
    
    xlabel(ax_cdf, filter_display_name, 'FontSize', 36, 'FontWeight', 'bold');
    ylabel(ax_cdf, 'Cumulative Percentage (%)', 'FontSize', 36, 'FontWeight', 'bold');
    title(ax_cdf, sprintf('CDF - %s (Left & Right)', filter_display_name), ...
        'FontSize', 38, 'FontWeight', 'bold');
    
    grid(ax_cdf, 'on');
    set(ax_cdf, 'FontSize', 32, 'LineWidth', 3);
    
    % Compact Stats Box (Southeast)
    stats_str = sprintf(['Left: Ver p50=%.1f, p99=%.1f | Non-ver p50=%.1f, p99=%.1f\n' ...
                        'Right: Ver p50=%.1f, p99=%.1f | Non-ver p50=%.1f, p99=%.1f'], ...
                        p_left_gt(1), p_left_gt(4), p_left_non_gt(1), p_left_non_gt(4), ...
                        p_right_gt(1), p_right_gt(4), p_right_non_gt(1), p_right_non_gt(4));
    text(ax_cdf, 0.98, 0.08, stats_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 22, 'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'black', ...
        'LineWidth', 2, 'Margin', 8);
    
    % Legend
    leg_cdf = legend(ax_cdf, 'FontSize', 28);
    set(leg_cdf, 'Units', 'normalized', 'Position', [0.62, 0.58, 0.30, 0.15]);
    
    ylim(ax_cdf, [0, 108]);
    hold(ax_cdf, 'off');
    
    % Save CDF figure
    save_filename_cdf = fullfile(output_dir, 'values', filter_base_name, sprintf('frame_%d_combined_cdf.png', frame_idx));
    saveas(fig_cdf, save_filename_cdf);
    fprintf('Saved combined CDF figure to: %s\n', save_filename_cdf);
    close(fig_cdf);
end
