% run_visualize_veridical.m
% Example script to visualize the complete veridical correspondence chain

% Set paths
csv_path = fullfile('..', 'outputs', 'veridical_debug_left.csv');

% Images from frame 0 (keyframe) and frame 1 (current frame)
kf_left_img  = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png/im0.png";
kf_right_img = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png/im1.png";
cf_left_img  = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png/im0.png";
cf_right_img = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png/im1.png";

% Check if files exist
if ~exist(csv_path, 'file')
    error('CSV file not found: %s', csv_path);
end

% Call visualization
visualize_veridical_chain(csv_path, kf_left_img, kf_right_img, cf_left_img, cf_right_img);

fprintf('\n========================================\n');
fprintf('Visualization Legend:\n');
fprintf('========================================\n');
fprintf('  A- (Red):     Keyframe left edge - original tracked edge\n');
fprintf('  B- (Cyan):    Keyframe right edges - stereo mates of A-\n');
fprintf('  A+ (Green):   Current frame left edges - veridical matches\n');
fprintf('  B+ (Blue):    Current frame right edges - from B- path\n');
fprintf('  B+ (Magenta): Current frame right edges - from A+ path\n');
fprintf('\n');
fprintf('CLICK on any edge to see:\n');
fprintf('  - Edge Index\n');
fprintf('  - Location (x, y)\n');
fprintf('  - Orientation (radians and degrees)\n');
fprintf('========================================\n');
