source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
stereo_pair_name = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
img_path = source_dataset_folder + dataset_sequence_path + stereo_pair_name + "/im0.png";
img_left = imread(img_path);

picked = [885.298, 167.764];

shifted_plus = [881.822, 164.17];
shifted_minus = [888.774, 171.358];

patch_size = 7;
half_patch = patch_size / 2;

patch_plus_x = [877.5797006627364, 878.2749163305763, 878.9701319984162, 879.665347666256, 880.3605633340959, 881.0557790019358, 881.7509946697757;
                878.29850186974, 878.9937175375799, 879.6889332054197, 880.3841488732597, 881.0793645410996, 881.7745802089395, 882.4697958767794;
                879.0173030767437, 879.7125187445836, 880.4077344124235, 881.1029500802633, 881.7981657481032, 882.4933814159431, 883.1885970837831;
                879.7361042837474, 880.4313199515873, 881.1265356194272, 881.821751287267, 882.5169669551069, 883.2121826229468, 883.9073982907867;
                880.454905490751, 881.150121158591, 881.8453368264309, 882.5405524942707, 883.2357681621106, 883.9309838299505, 884.6261994977904;
                881.1737066977547, 881.8689223655946, 882.5641380334345, 883.2593537012743, 883.9545693691143, 884.6497850369542, 885.3450007047941;
                881.8925079047584, 882.5877235725983, 883.2829392404382, 883.978154908278, 884.6733705761179, 885.3685862439578, 886.0638019117977];

patch_plus_y = [164.0989867136603, 164.817787920664, 165.5365891276676, 166.2553903346713, 166.974191541675, 167.6929927486787, 168.4117939556824;
                163.4037710458204, 164.1225722528241, 164.8413734598278, 165.5601746668314, 166.2789758738351, 166.9977770808388, 167.7165782878425;
                162.7085553779805, 163.4273565849842, 164.1461577919879, 164.8649589989915, 165.5837602059952, 166.3025614129989, 167.0213626200026;
                162.0133397101406, 162.7321409171443, 163.450942124148, 164.1697433311517, 164.8885445381553, 165.607345745159, 166.3261469521627;
                161.3181240423008, 162.0369252493044, 162.7557264563081, 163.4745276633118, 164.1933288703154, 164.9121300773191, 165.6309312843228;
                160.6229083744609, 161.3417095814645, 162.0605107884682, 162.7793119954719, 163.4981132024756, 164.2169144094792, 164.9357156164829;
                159.927692706621, 160.6464939136247, 161.3652951206283, 162.084096327632, 162.8028975346357, 163.5216987416394, 164.240499948643];

patch_minus_x = [884.5318573411352, 885.227073008975, 885.9222886768149, 886.6175043446548, 887.3127200124947, 888.0079356803345, 888.7031513481744;
                 885.2506585481387, 885.9458742159786, 886.6410898838185, 887.3363055516585, 888.0315212194984, 888.7267368873382, 889.4219525551781;
                 885.9694597551425, 886.6646754229823, 887.3598910908222, 888.0551067586621, 888.750322426502, 889.4455380943418, 890.1407537621818;
                 886.6882609621462, 887.383476629986, 888.0786922978259, 888.7739079656658, 889.4691236335057, 890.1643393013455, 890.8595549691854;
                 887.4070621691498, 888.1022778369897, 888.7974935048296, 889.4927091726695, 890.1879248405094, 890.8831405083492, 891.5783561761891;
                 888.1258633761535, 888.8210790439933, 889.5162947118332, 890.2115103796731, 890.9067260475131, 891.601941715353, 892.2971573831928;
                 888.8446645831572, 889.539880250997, 890.2350959188369, 890.9303115866768, 891.6255272545167, 892.3207429223565, 893.0159585901964];

patch_minus_y = [171.286998783697, 172.0057999907007, 172.7246011977044, 173.4434024047081, 174.1622036117117, 174.8810048187154, 175.5998060257191;
                 170.5917831158571, 171.3105843228608, 172.0293855298645, 172.7481867368682, 173.4669879438719, 174.1857891508755, 174.9045903578792;
                 169.8965674480173, 170.6153686550209, 171.3341698620246, 172.0529710690283, 172.771772276032, 173.4905734830356, 174.2093746900393;
                 169.2013517801774, 169.920152987181, 170.6389541941847, 171.3577554011884, 172.0765566081921, 172.7953578151958, 173.5141590221994;
                 168.5061361123375, 169.2249373193412, 169.9437385263448, 170.6625397333485, 171.3813409403522, 172.1001421473559, 172.8189433543595;
                 167.8109204444976, 168.5297216515013, 169.2485228585049, 169.9673240655086, 170.6861252725123, 171.404926479516, 172.1237276865197;
                 167.1157047766577, 167.8345059836614, 168.5533071906651, 169.2721083976687, 169.9909096046724, 170.7097108116761, 171.4285120186798];

if size(img_left, 3) == 3
    img_gray = rgb2gray(img_left);
else
    img_gray = img_left;
end
fprintf('Image size: %d x %d (rows x cols)\n', size(img_gray, 1), size(img_gray, 2));

[rows, cols] = size(img_gray);
patch_plus_values = zeros(size(patch_plus_x));
patch_minus_values = zeros(size(patch_minus_x));
fprintf('Extracting patch (+) values...\n');
for i = 1:size(patch_plus_x, 1)
    for j = 1:size(patch_plus_x, 2)
        x = patch_plus_x(i, j);
        y = patch_plus_y(i, j);
        patch_plus_values(i, j) = bilinear_interp_custom(img_gray, x, y);
    end
end

% Extract patch (-) values
fprintf('Extracting patch (-) values...\n');
for i = 1:size(patch_minus_x, 1)
    for j = 1:size(patch_minus_x, 2)
        x = patch_minus_x(i, j);
        y = patch_minus_y(i, j);
        patch_minus_values(i, j) = bilinear_interp_custom(img_gray, x, y);
    end
end

% Display statistics
fprintf('\n=== MATLAB PATCH EXTRACTION RESULTS ===\n');
fprintf('Patch (+) - Min: %.2f, Max: %.2f, Mean: %.2f, Std: %.2f\n', ...
        min(patch_plus_values(:)), max(patch_plus_values(:)), ...
        mean(patch_plus_values(:)), std(patch_plus_values(:)));
fprintf('Patch (-) - Min: %.2f, Max: %.2f, Mean: %.2f, Std: %.2f\n', ...
        min(patch_minus_values(:)), max(patch_minus_values(:)), ...
        mean(patch_minus_values(:)), std(patch_minus_values(:)));

plus_zeros = sum(patch_plus_values(:) == 0);
minus_zeros = sum(patch_minus_values(:) == 0);
fprintf('Patch (+) zero values (out of bounds): %d/%d\n', plus_zeros, numel(patch_plus_values));
fprintf('Patch (-) zero values (out of bounds): %d/%d\n', minus_zeros, numel(patch_minus_values));

% Create visualizations
scale_factor = 50; % Make each pixel 50x50 for visibility

% Convert to uint8 for display
patch_plus_uint8 = uint8(patch_plus_values);
patch_minus_uint8 = uint8(patch_minus_values);

% Resize for better visualization
patch_plus_large = imresize(patch_plus_uint8, scale_factor, 'nearest');
patch_minus_large = imresize(patch_minus_uint8, scale_factor, 'nearest');

% Create figure for patches
figure('Position', [300, 300, 800, 400]);

subplot(1, 3, 1);
imshow(patch_plus_large);
title(sprintf('Patch (+) - Rotated\nMean: %.1f, Std: %.1f', mean(patch_plus_values(:)), std(patch_plus_values(:))));
colorbar;

subplot(1, 3, 2);
imshow(patch_minus_large);
title(sprintf('Patch (-) - Rotated\nMean: %.1f, Std: %.1f', mean(patch_minus_values(:)), std(patch_minus_values(:))));
colorbar;

% Side-by-side comparison
patch_comparison = [patch_plus_large, 128*ones(size(patch_plus_large, 1), 20), patch_minus_large];
subplot(1, 3, 3);
imshow(patch_comparison);
title('Side-by-Side Comparison');

if plus_zeros == 0 && minus_zeros == 0
    correlation_coeff = corrcoef(patch_plus_values(:), patch_minus_values(:));
    fprintf('Correlation coefficient: %.4f\n', correlation_coeff(1,2));
else
    fprintf('Cannot compute correlation - patches contain out-of-bounds values\n');
end

% Save the patches
imwrite(patch_plus_large, 'matlab_extracted_patch_plus.png');
imwrite(patch_minus_large, 'matlab_extracted_patch_minus.png');
imwrite(patch_comparison, 'matlab_extracted_patch_comparison.png');

fprintf('\nSaved files:\n');
fprintf('- matlab_extracted_patch_plus.png\n');
fprintf('- matlab_extracted_patch_minus.png\n');
fprintf('- matlab_extracted_patch_comparison.png\n');

% Display patch values as matrices
fprintf('\nPatch (+) values:\n');
disp(patch_plus_values);
fprintf('\nPatch (-) values:\n');
disp(patch_minus_values);
% Add this to the end of your test.m file

% Create visualization of edge and shifted points on original image
figure('Position', [100, 100, 1200, 800]);
imshow(img_left);
hold on;

% Plot the picked edge point (green circle)
plot(picked(1), picked(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'green', 'LineWidth', 3);

% Plot the shifted points
plot(shifted_plus(1), shifted_plus(2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'blue', 'LineWidth', 2);
plot(shifted_minus(1), shifted_minus(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'red', 'LineWidth', 2);

% Draw lines connecting picked point to shifted points
plot([picked(1), shifted_plus(1)], [picked(2), shifted_plus(2)], 'b-', 'LineWidth', 2);
plot([picked(1), shifted_minus(1)], [picked(2), shifted_minus(2)], 'r-', 'LineWidth', 2);

% Calculate edge orientation from shifted points
edge_vector = shifted_minus - shifted_plus;
edge_orientation = atan2(edge_vector(2), edge_vector(1));

% Draw edge orientation line (yellow)
line_length = 40;
end_point_x = picked(1) + line_length * cos(edge_orientation);
end_point_y = picked(2) + line_length * sin(edge_orientation);
plot([picked(1), end_point_x], [picked(2), end_point_y], 'y-', 'LineWidth', 4);

% Draw patch boundaries using the actual coordinates
% For patch (+) - show min/max bounds
plus_x_bounds = [min(patch_plus_x(:)), max(patch_plus_x(:))];
plus_y_bounds = [min(patch_plus_y(:)), max(patch_plus_y(:))];
patch_width_plus = plus_x_bounds(2) - plus_x_bounds(1);
patch_height_plus = plus_y_bounds(2) - plus_y_bounds(1);

rectangle('Position', [plus_x_bounds(1), plus_y_bounds(1), patch_width_plus, patch_height_plus], ...
         'EdgeColor', 'blue', 'LineWidth', 2, 'LineStyle', '--');

% For patch (-) - show min/max bounds
minus_x_bounds = [min(patch_minus_x(:)), max(patch_minus_x(:))];
minus_y_bounds = [min(patch_minus_y(:)), max(patch_minus_y(:))];
patch_width_minus = minus_x_bounds(2) - minus_x_bounds(1);
patch_height_minus = minus_y_bounds(2) - minus_y_bounds(1);

rectangle('Position', [minus_x_bounds(1), minus_y_bounds(1), patch_width_minus, patch_height_minus], ...
         'EdgeColor', 'red', 'LineWidth', 2, 'LineStyle', '--');

% Show the actual patch coordinate points as small dots
plot(patch_plus_x(:), patch_plus_y(:), 'b.', 'MarkerSize', 4);
plot(patch_minus_x(:), patch_minus_y(:), 'r.', 'MarkerSize', 4);

% Add legend
legend({'Edge Point', 'Shifted (+)', 'Shifted (-)', 'Line to (+)', 'Line to (-)', ...
        'Edge Orientation', 'Patch (+) Bounds', 'Patch (-) Bounds', ...
        'Patch (+) Points', 'Patch (-) Points'}, ...
       'Location', 'best', 'FontSize', 10);

% Add title and labels
title('Edge Point with Rotated Patch Extraction', 'FontSize', 16);
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);

% Add text annotations
text(picked(1) + 5, picked(2) - 5, sprintf('Edge (%.1f, %.1f)', picked(1), picked(2)), ...
     'Color', 'green', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'white');
text(shifted_plus(1) + 5, shifted_plus(2) - 5, sprintf('(+) (%.1f, %.1f)', shifted_plus(1), shifted_plus(2)), ...
     'Color', 'blue', 'FontSize', 10, 'BackgroundColor', 'white');
text(shifted_minus(1) + 5, shifted_minus(2) + 10, sprintf('(-) (%.1f, %.1f)', shifted_minus(1), shifted_minus(2)), ...
     'Color', 'red', 'FontSize', 10, 'BackgroundColor', 'white');

% Set axis to show the region around the edge
axis_margin = 80;
xlim([picked(1) - axis_margin, picked(1) + axis_margin]);
ylim([picked(2) - axis_margin, picked(2) + axis_margin]);

% Add grid for better visualization
grid on;
grid minor;

hold off;

% Save the image visualization
saveas(gcf, 'edge_and_rotated_patches_visualization.png');
fprintf('Saved edge visualization: edge_and_rotated_patches_visualization.png\n');

% Print detailed information
fprintf('\n=== EDGE AND PATCH INFORMATION ===\n');
fprintf('Image size: %d x %d (rows x cols)\n', size(img_gray, 1), size(img_gray, 2));
fprintf('Edge Point: (%.3f, %.3f)\n', picked(1), picked(2));
fprintf('Shifted (+): (%.3f, %.3f)\n', shifted_plus(1), shifted_plus(2));
fprintf('Shifted (-): (%.3f, %.3f)\n', shifted_minus(1), shifted_minus(2));

% Calculate distances
dist_plus = sqrt((picked(1) - shifted_plus(1))^2 + (picked(2) - shifted_plus(2))^2);
dist_minus = sqrt((picked(1) - shifted_minus(1))^2 + (picked(2) - shifted_minus(2))^2);
fprintf('Distance to (+): %.3f pixels\n', dist_plus);
fprintf('Distance to (-): %.3f pixels\n', dist_minus);

% Edge orientation
fprintf('Edge Orientation: %.2f degrees\n', rad2deg(edge_orientation));

% Patch bounds
fprintf('\nPatch (+) bounds: X[%.1f, %.1f], Y[%.1f, %.1f]\n', ...
        plus_x_bounds(1), plus_x_bounds(2), plus_y_bounds(1), plus_y_bounds(2));
fprintf('Patch (-) bounds: X[%.1f, %.1f], Y[%.1f, %.1f]\n', ...
        minus_x_bounds(1), minus_x_bounds(2), minus_y_bounds(1), minus_y_bounds(2));

% Check if patches are within image bounds
plus_in_bounds = plus_x_bounds(1) >= 1 && plus_x_bounds(2) <= cols && ...
                 plus_y_bounds(1) >= 1 && plus_y_bounds(2) <= rows;
minus_in_bounds = minus_x_bounds(1) >= 1 && minus_x_bounds(2) <= cols && ...
                  minus_y_bounds(1) >= 1 && minus_y_bounds(2) <= rows;

fprintf('Patch (+) fully in bounds: %s\n', mat2str(plus_in_bounds));
fprintf('Patch (-) fully in bounds: %s\n', mat2str(minus_in_bounds));

if ~plus_in_bounds
    fprintf('WARNING: Patch (+) coordinates go outside image bounds!\n');
end
if ~minus_in_bounds
    fprintf('WARNING: Patch (-) coordinates go outside image bounds!\n');
end
function val = bilinear_interp_custom(img, x, y)
    [rows, cols] = size(img);
    
    % Bounds check (same as your C++ bounds)
    if x < 1 || x >= cols-1 || y < 1 || y >= rows-1
        val = 0;
        return;
    end
    
    % Get integer coordinates
    x1 = floor(x);
    y1 = floor(y);
    x2 = x1 + 1;
    y2 = y1 + 1;
    
    % Get fractional parts
    dx = x - x1;
    dy = y - y1;
    
    % Convert to MATLAB indexing (1-based)
    x1 = x1 + 1;
    x2 = x2 + 1;
    y1 = y1 + 1;
    y2 = y2 + 1;
    
    % Check bounds again for MATLAB indexing
    if x2 > cols || y2 > rows
        val = 0;
        return;
    end
    
    % Get pixel values (MATLAB uses (row, col) indexing)
    val11 = double(img(y1, x1));
    val12 = double(img(y2, x1));
    val21 = double(img(y1, x2));
    val22 = double(img(y2, x2));
    
    % Bilinear interpolation
    val1 = val11 * (1 - dx) + val21 * dx;
    val2 = val12 * (1 - dx) + val22 * dx;
    val = val1 * (1 - dy) + val2 * dy;
end