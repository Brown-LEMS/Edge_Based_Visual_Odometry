function ncc_score = compute_ncc(patch1, patch2)
    % Ensure patches are the same size
    if ~isequal(size(patch1), size(patch2))
        error('Patches must be the same size');
    end
    
    % Convert to double if needed
    patch1 = double(patch1);
    patch2 = double(patch2);
    
    % Compute means
    mean1 = mean(patch1(:));
    mean2 = mean(patch2(:));
    
    % Zero-mean patches
    patch1_zm = patch1 - mean1;
    patch2_zm = patch2 - mean2;
    
    % Compute variances
    var1 = sum(patch1_zm(:).^2);
    var2 = sum(patch2_zm(:).^2);
    
    % Handle zero variance case
    if var1 < 1e-10 || var2 < 1e-10
        ncc_score = -1.0;  % Return negative correlation
        return;
    end
    
    % Compute NCC
    numerator = sum(patch1_zm(:) .* patch2_zm(:));
    denominator = sqrt(var1 * var2);
    
    ncc_score = numerator / denominator;
    
    % Clamp to [-1, 1] range due to numerical precision
    ncc_score = max(-1.0, min(1.0, ncc_score));
end