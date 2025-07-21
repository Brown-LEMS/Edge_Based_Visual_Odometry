#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

// #include "./test_third_order_edges.hpp"
#include "../include/utility.h"
#include "../include/definitions.h"

template<typename T>
T Uniform_Random_Number_Generator(T range_from, T range_to) {
    std::random_device                                          rand_dev;
    std::mt19937                                                rng(rand_dev());
    std::uniform_int_distribution<std::mt19937::result_type>    distr(range_from, range_to);
    return distr(rng);
}

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points( const Edge edgel )
{
    double shifted_x1 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel.orientation));
    double shifted_y1 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel.orientation));
    double shifted_x2 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel.orientation));
    double shifted_y2 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel.orientation));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

void get_patch_on_one_edge_side( cv::Point2d shifted_point, double theta, \
                                 cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, \
                                 cv::Mat &patch_val, const cv::Mat img ) 
{
    int half_patch_size = floor(PATCH_SIZE / 2);
    for (int i = -half_patch_size; i <= half_patch_size; i++) {
        for (int j = -half_patch_size; j <= half_patch_size; j++) {
            //> get the rotated coordinate
            cv::Point2d rotated_point(cos(theta)*(i) - sin(theta)*(j) + shifted_point.x, sin(theta)*(i) + cos(theta)*(j) + shifted_point.y);
            patch_coord_x.at<int>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<int>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            //> get the image intensity of the rotated coordinate
            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

void f_TEST_NCC() 
{
    std::string source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
    std::string dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
    std::string stereo_pair_name = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
    std::string img_path = source_dataset_folder + dataset_sequence_path + stereo_pair_name + "/im0.png";
    
    const cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    const int img_height = img.rows;
    const int img_width  = img.cols;
    std::cout << "image dimension: (" << img_height << ", " << img_width << ")" << std::endl;

    ThirdOrderEdgeDetectionCPU toed_engine(img_height, img_width);
    
    LOG_INFO("Running third-order edge detector...");
    std::vector<Edge> toed_edges = get_third_order_edges_(img, toed_engine);

    Edge target_edge;
    while (true) {
        int rand_edge_idx = Uniform_Random_Number_Generator< int >(0, toed_edges.size()-1);
        target_edge = toed_edges[rand_edge_idx]; 
        if (fabs(fabs(target_edge.orientation) - fabs(M_PI / 4.0)) < 0.02)
            break;
    }
    
    std::cout << "Picked edge: (" << target_edge.location.x << ", " << target_edge.location.y << ", " << target_edge.orientation << ")" << std::endl;
    std::cout << "Orientation in degree: " << rad_to_deg<double>(target_edge.orientation) << std::endl;

    std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points( target_edge );

    cv::Mat patch_coord_x_plus  = cv::Mat_<int>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_plus  = cv::Mat_<int>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_minus = cv::Mat_<int>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_minus = cv::Mat_<int>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

    //> get the patches on the two sides of the edge
    get_patch_on_one_edge_side( shifted_points.first,  target_edge.orientation, patch_coord_x_plus,  patch_coord_y_plus,  patch_plus,  img );
    get_patch_on_one_edge_side( shifted_points.second, target_edge.orientation, patch_coord_x_minus, patch_coord_y_minus, patch_minus, img );

    std::cout << "Shifted point (+) location: (" << shifted_points.first.x << ", " << shifted_points.first.y << ")" << std::endl;
    std::cout << "Patch (+) coordinates: " << std::endl;
    std::cout << patch_coord_x_plus << std::endl;
    std::cout << patch_coord_y_plus << std::endl;

    std::cout << "Shifted point (-) location: (" << shifted_points.second.x << ", " << shifted_points.second.y << ")" << std::endl;
    std::cout << "Patch (-) coordinates: " << std::endl;
    std::cout << patch_coord_x_minus << std::endl;
    std::cout << patch_coord_y_minus << std::endl;

    if (patch_plus.type() != CV_32F) {
        patch_plus.convertTo(patch_plus, CV_32F);
    }
    if (patch_minus.type() != CV_32F) {
        patch_minus.convertTo(patch_minus, CV_32F);
    }
}