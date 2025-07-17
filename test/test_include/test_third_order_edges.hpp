#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../include/toed/cpu_toed.hpp"
#include "../include/utility.h"
#include "../include/definitions.h"

void f_TEST_TOED() 
{
    std::string source_dataset_folder = "/gpfs/data/bkimia/Datasets/ETH3D/";
    std::string dataset_sequence_path = "stereo/delivery_area/stereo_pairs/";
    std::string stereo_pair_name = "images_rig_cam4-1477843917481127523.png-images_rig_cam5-1477843917481127523.png";
    std::string img_path = source_dataset_folder + dataset_sequence_path + stereo_pair_name + "/im0.png";
    
    const cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    const int img_height = img.rows;
    const int img_width  = img.cols;

    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(img_height, img_width));
    
    LOG_INFO("Running third-order edge detector...");
    TOED->get_Third_Order_Edges(img);
    std::vector<Edge> edges = TOED->toed_edges;
    std::cout << "Number of total edges = " << edges.size() << std::endl;

}