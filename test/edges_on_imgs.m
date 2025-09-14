
source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
stereo_pair_name = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
img_path = source_dataset_folder + dataset_sequence_path + stereo_pair_name + "/im0.png";
img_left = imread(img_path);
img_path = source_dataset_folder + dataset_sequence_path + stereo_pair_name + "/im1.png";
img_right = imread(img_path);

toed_left = importdata("toed.txt");

dirs_cos = cos(toed_left(:,3)); 
dirs_sin = sin(toed_left(:,3));
dir_vecs = [dirs_cos, dirs_sin];

figure(1);
imshow(img_left); hold on;
plot(toed_left(:,1), toed_left(:,2), 'c.'); hold on;
quiver(toed_left(:,1), toed_left(:,2), dir_vecs(:,1), dir_vecs(:,2), 0, 'c-', 'LineWidth', 1.2);


