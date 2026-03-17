

kf_left_img  = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png/im0.png";
cf_left_img  = "/oscar/data/bkimia/Datasets/ETH3D/stereo/delivery_area/stereo_pairs/images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png/im0.png";




plot_epipolar(kf_left_img,'../ep_lines.txt','../ep_edges.txt', cf_left_img)