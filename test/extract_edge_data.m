% filepath: /oscar/data/bkimia/jhan192/jue_dev_ebvo/Edge_Based_Visual_Odometry/test/extract_edge_data.m
function extract_edge_data(csv_file, prev_edge_idx)
    % Extract edge data for visualization from CSV file
    % Inputs:
    %   csv_file - path to the CSV file with edge matching results
    %   prev_edge_idx - index of the previous edge to extract data for
    
    % Read the CSV file
    data = readtable(csv_file);
    
    % Filter rows for the specified previous edge index
    prev_edge_data = data(data.prev_edge_idx == prev_edge_idx, :);
    
    if isempty(prev_edge_data)
        fprintf('No data found for previous edge index %d\n', prev_edge_idx);
        return;
    end
    
    % Extract previous edge data (should be the same for all rows)
    prev_x = prev_edge_data.prev_x(1);
    prev_y = prev_edge_data.prev_y(1);
    prev_orientation = prev_edge_data.prev_orientation(1);
    
    % Create prev.txt with format: x y orientation
    prev_data = [prev_x, prev_y, prev_orientation];
    dlmwrite('prev.txt', prev_data, 'precision', '%.3f');
    fprintf('Previous edge saved to prev.txt\n');
    
    % Extract GT edge (there should be only one)
    gt_row = prev_edge_data(strcmp(prev_edge_data.candidate_type, 'GT'), :);
    
    if isempty(gt_row)
        fprintf('No GT edge found for previous edge index %d\n', prev_edge_idx);
        return;
    end
    
    % Extract candidate edges
    candidate_rows = prev_edge_data(strcmp(prev_edge_data.candidate_type, 'Candidate'), :);
    
    if isempty(candidate_rows)
        fprintf('No candidate edges found for previous edge index %d\n', prev_edge_idx);
    end
    
    % Create gt.txt with format: x y orientation
    gt_data = [gt_row.gt_x, gt_row.gt_y, gt_row.candidate_orientation];
    dlmwrite('gt.txt', gt_data, 'precision', '%.3f');
    fprintf('Ground truth edge saved to gt.txt\n');
    
    % Create ncc.txt with format: x y orientation
    ncc_data = [candidate_rows.candidate_x, candidate_rows.candidate_y, candidate_rows.candidate_orientation];
    dlmwrite('ncc.txt', ncc_data, 'precision', '%.3f');
    fprintf('Candidate edges saved to ncc.txt\n');
    
    % Print summary
    fprintf('Found %d candidate edges for previous edge %d\n', height(candidate_rows), prev_edge_idx);
    fprintf('Previous edge location: (%.3f, %.3f)\n', prev_x, prev_y);
    fprintf('GT edge location: (%.3f, %.3f)\n', gt_row.gt_x, gt_row.gt_y);
end