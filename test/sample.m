% Example runner for KF/CF right-image visualization for edge 48660
csv_path = fullfile('..','outputs','kf_cf_projection_edge_48660_right.csv');
cf_img   = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png/im1.png";
kf_img   = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png/im1.png";
visualize_kf_cf_projection(csv_path, kf_img, cf_img);