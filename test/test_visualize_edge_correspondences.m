clear;
close all;

source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
last_keyframe_stereo_folder = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
current_frame_stereo_folder = "images_rig_cam4-1477843917554855523.png-images_rig_cam5-1477843917554855523.png";
KF_img_path = source_dataset_folder + dataset_sequence_path + last_keyframe_stereo_folder + "/im0.png";
CF_img_path = source_dataset_folder + dataset_sequence_path + current_frame_stereo_folder + "/im0.png";
kf_img = imread(KF_img_path);
cf_img = imread(CF_img_path);

gt_kf_cf_edge_pairs_kf_filename = fullfile(pwd, "outputs", "kf_cf_gt_edge_pairs_KF.txt");
gt_kf_cf_edge_pairs_cf_filename = fullfile(pwd, "outputs", "kf_cf_gt_edge_pairs_CF.txt");
list_of_edge_pairs_KF = importdata(gt_kf_cf_edge_pairs_kf_filename);
list_of_edge_pairs_CF = importdata(gt_kf_cf_edge_pairs_cf_filename);

%> randomly choose N edge correspondences
N = 20;
rand_edge_index = randperm(size(list_of_edge_pairs_KF,1), N);
picked_edge_kf_gt_pairs = list_of_edge_pairs_KF(rand_edge_index, :);
picked_edge_indices = picked_edge_kf_gt_pairs(:,1);

cf_edge_sets = cell(N,1);
for i = 1:length(picked_edge_indices)
    cf_list_indices = find(list_of_edge_pairs_CF(:,1) == picked_edge_indices(i));
    cf_edge_sets{i} = list_of_edge_pairs_CF(cf_list_indices,:);
end
colors = lines(length(rand_edge_index)); 

%> Show the correspondences between left edge of keyframe and the GT
%  propjections on the current frame as well as a set of close-to-GT left
%  edges of the current frame
figure(1);
stack_imgs = [kf_img; cf_img];
imshow(stack_imgs); hold on;
for i = 1:length(rand_edge_index)
    plot(picked_edge_kf_gt_pairs(i,2)+1, picked_edge_kf_gt_pairs(i,3)+1, "Color", colors(i,:), 'Marker', 'o', 'LineWidth', 2);
    hold on;
    quiver(picked_edge_kf_gt_pairs(i,2)+1, picked_edge_kf_gt_pairs(i,3)+1, ...
           cos(picked_edge_kf_gt_pairs(i,4)), sin(picked_edge_kf_gt_pairs(i,4)), 0, 'Color', colors(i,:), 'LineWidth', 2);
    hold on;
    plot(picked_edge_kf_gt_pairs(i,5)+1, picked_edge_kf_gt_pairs(i,6)+1+size(cf_img,1), "Color", colors(i,:), 'Marker', 'o', 'LineWidth', 2);
    hold on;
    for j = 1:size(cf_edge_sets{i},1)
        plot(cf_edge_sets{i}(j,2)+1, cf_edge_sets{i}(j,3)+1+size(cf_img,1), "Color", colors(i,:), 'Marker', '+', 'LineWidth', 2);
        hold on;
        quiver(cf_edge_sets{i}(j,2)+1, cf_edge_sets{i}(j,3)+1+size(cf_img,1), ...
               cos(cf_edge_sets{i}(j,4)), sin(cf_edge_sets{i}(j,4)), 0, 'Color', colors(i,:), 'LineWidth', 2);
        hold on;
    end
end
title("(Top) Keyframe. (Bottom) Current frame.");
