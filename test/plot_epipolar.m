function plot_epipolar(img_path_kf, epiline_file, edge_file, img_path_cf, epiline_file2)
% plot_epipolar(img_path_kf, epiline_file, edge_file, img_path_cf, epiline_file2)
% - img_path_kf: path to the KF (keyframe) image to overlay KF edges on ('' to skip)
% - epiline_file: file containing epipolar lines (KF->CF) (CSV or debug log)
% - edge_file: file containing KF edge points (CSV or debug log)
% - img_path_cf: optional CF (current frame) image path to plot epipolar lines on
% - epiline_file2: optional second epipolar-lines file (for another frame)

 if nargin < 1, img_path_kf = ''; end
if nargin < 2, error('Need epiline_file'); end
if nargin < 3, edge_file = ''; end
 if nargin < 4, img_path_cf = ''; end
 if nargin < 5, epiline_file2 = ''; end

% Try to read image
 % Try to read KF image
 img_kf_loaded = false;
 if ~isempty(img_path_kf) && exist(img_path_kf, 'file')
    I_kf = imread(img_path_kf);
    img_kf_loaded = true;
    sz_kf = size(I_kf); img_w_kf = sz_kf(2); img_h_kf = sz_kf(1);
    fig_kf = figure('Name','KF Image'); imshow(I_kf); hold on;
 else
     img_w_kf = 1280; img_h_kf = 720; % default canvas
    fig_kf = figure('Name','KF Image'); axis([1 img_w_kf 1 img_h_kf]); set(gca,'YDir','reverse'); hold on;
 end

 % Try to read CF image (for plotting epipolar lines)
 img_cf_loaded = false;
 if ~isempty(img_path_cf) && exist(img_path_cf, 'file')
    I_cf = imread(img_path_cf);
    img_cf_loaded = true;
    sz_cf = size(I_cf); img_w_cf = sz_cf(2); img_h_cf = sz_cf(1);
    fig_cf = figure('Name','CF Image'); % create handle now, will show image when plotting
 else
     img_w_cf = img_w_kf; img_h_cf = img_h_kf; % fallback to KF dims
end

[a,b,c] = deal([]);
% Try numeric file: whitespace or CSV (3 cols)
T = try_read_matrix(epiline_file, 3);
if ~isempty(T)
    a = T(:,1); b = T(:,2); c = T(:,3);
else
    % try parsing debug log lines like "Epipolar line coeffs: a b c"
    fid = fopen(epiline_file,'r');
    if fid ~= -1
        txt = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
        fclose(fid);
        lines = txt{1};
        for i=1:numel(lines)
            s = strtrim(lines{i});
            idx = strfind(s, 'Epipolar line coeffs:');
            if ~isempty(idx)
                vals = sscanf(s(idx+length('Epipolar line coeffs:'):end), '%f');
                if numel(vals) >= 3
                    a(end+1,1)=vals(1); b(end+1,1)=vals(2); c(end+1,1)=vals(3);
                end
            end
        end
    end
end

% Read optional second set of epipolar lines
 [a2,b2,c2] = deal([]);
 if ~isempty(epiline_file2)
     T = try_read_matrix(epiline_file2, 3);
     if ~isempty(T)
         a2 = T(:,1); b2 = T(:,2); c2 = T(:,3);
     else
         fid = fopen(epiline_file2,'r');
         if fid ~= -1
             txt = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
             fclose(fid);
             lines2 = txt{1};
             for i=1:numel(lines2)
                 s = strtrim(lines2{i});
                 idx = strfind(s, 'Epipolar line coeffs:');
                 if ~isempty(idx)
                     vals = sscanf(s(idx+length('Epipolar line coeffs:'):end), '%f');
                     if numel(vals) >= 3
                         a2(end+1,1)=vals(1); b2(end+1,1)=vals(2); c2(end+1,1)=vals(3);
                     end
                 end
             end
         end
     end
 end

% Read edge points
edges = [];
T2 = try_read_matrix(edge_file, 2);
if ~isempty(T2)
    edges = T2(:,1:2);
else
    % parse debug log lines like "Edge 0: location = (x, y)"
    if exist(edge_file,'file')
        fid = fopen(edge_file,'r');
        txt = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
        fclose(fid);
        lines = txt{1};
        for i=1:numel(lines)
            s = lines{i};
            idx = strfind(s, 'Edge');
            if ~isempty(idx) && contains(s, 'location')
                nums = regexp(s, '\(([-\d.+eE\s,]+)\)', 'tokens');
                if ~isempty(nums)
                    xy = sscanf(nums{1}{1}, '%f,%f');
                    if numel(xy)==2, edges(end+1,:) = xy'; end
                end
            end
        end
    end
end

% Randomly sample up to 5 edges for plotting to avoid clutter
max_samples = 5;
selected_idx = [];
Nedges = size(edges,1);
if Nedges > 0
    ns = min(max_samples, Nedges);
    rng('shuffle');
    selected_idx = randperm(Nedges, ns);
    sel_edges = edges(selected_idx, :);
end

% Plot epipolar lines
 % Plot epipolar lines on CF image (light color)
 if ~isempty(a)
     num_lines = numel(a);
     pale_red = [1.0, 0.6, 0.6];
     % Determine which lines to draw: prefer sampled KF edges
     if exist('selected_idx','var') && ~isempty(selected_idx)
         line_indices = selected_idx(selected_idx <= num_lines);
     else
         line_indices = [];
     end
     if isempty(line_indices)
         nl = min(max_samples, num_lines);
         rng('shuffle');
         line_indices = randperm(num_lines, nl);
     end
     % Plot on CF image (open CF figure)
    if img_cf_loaded
        figure(fig_cf); imshow(I_cf); hold on;
    else
        figure(fig_cf); axis([1 img_w_cf 1 img_h_cf]); set(gca,'YDir','reverse'); hold on;
    end
     for ii = 1:numel(line_indices)
         k = line_indices(ii);
         ak = a(k); bk = b(k); ck = c(k);
         if abs(bk) > 1e-8
             x = [1, img_w_cf];
             y = (-ak*x - ck)/bk;
             plot(x, y, '-', 'Color', pale_red, 'LineWidth', 0.9);
         else
             if abs(ak) > 1e-8
                 x0 = -ck/ak;
                 plot([x0 x0], [1 img_h_cf], '-', 'Color', pale_red, 'LineWidth', 0.9);
             end
         end
     end
 end

% Secondary set (next frame) in pale blue
if ~isempty(a2)
    num_lines2 = numel(a2);
    pale_blue = [0.6, 0.8, 1.0];
    nl2 = min(max_samples, num_lines2);
    rng(1); % deterministic pick for second set
    line_indices2 = randperm(num_lines2, nl2);
    for ii = 1:numel(line_indices2)
        k = line_indices2(ii);
        ak = a2(k); bk = b2(k); ck = c2(k);
        if abs(bk) > 1e-8
            x = [1, img_w_cf];
            y = (-ak*x - ck)/bk;
            plot(x, y, '--', 'Color', pale_blue, 'LineWidth', 0.9);
        else
            if abs(ak) > 1e-8
                x0 = -ck/ak;
                plot([x0 x0], [1 img_h_cf], '--', 'Color', pale_blue, 'LineWidth', 0.9);
            end
        end
    end
end

% Now plot sampled edges on top so they remain visible
% Plot sampled edges on the KF figure (so edges appear on KF)
if exist('sel_edges','var') && ~isempty(sel_edges)
    if exist('fig_kf','var')
        figure(fig_kf); hold on;
    end
    plot(sel_edges(:,1), sel_edges(:,2), 'bo', 'MarkerSize', 8, 'LineWidth', 1.2, 'MarkerFaceColor', 'b');
end

title(sprintf('Epipolar lines (%d) and edges (%d)', numel(a), size(edges,1)));
hold off;
end

function M = try_read_matrix(fn, cols)
M = [];
if ~exist(fn,'file'), return; end
% try textscan numeric
fid = fopen(fn,'r');
if fid == -1, return; end
data = textscan(fid, repmat('%f',1,cols), 'Delimiter', ' ,\t', 'CollectOutput', 1);
fclose(fid);
if ~isempty(data) && ~isempty(data{1}) && size(data{1},2)==cols
    M = data{1};
end
end