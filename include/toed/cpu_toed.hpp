#ifndef CPU_TOED_HPP
#define CPU_TOED_HPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

#include "indices.hpp"
#include <omp.h>
#include <opencv2/opencv.hpp>

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Chien  25-02-08    Imported from the original third-order edge detector.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

class ThirdOrderEdgeDetectionCPU {

  typedef std::shared_ptr<ThirdOrderEdgeDetectionCPU> Ptr;
  int img_height;
  int img_width;
  int interp_img_height;
  int interp_img_width;
  int kernel_sz;
  int shifted_kernel_sz;
  int g_sig;
  int interp_n;

  double *img;
  double *Ix, *Iy;
  double *I_grad_mag;
  double *I_orient;

  double *subpix_pos_x_map;         //> store x of subpixel location --
  double *subpix_pos_y_map;         //> store y of subpixel location --
  double *subpix_grad_mag_map;      //> store subpixel gradient magnitude --

public:

  double *subpix_edge_pts_final;    //> a list of final edge points with all information (Nx4 array, where N is the number of third-order edges)
  int edge_pt_list_idx;
  int num_of_edge_data;
  int omp_threads;

  //> timings
  double time_conv, time_nms;

  ThirdOrderEdgeDetectionCPU(int, int);
  ~ThirdOrderEdgeDetectionCPU();

  //> member functions
  void get_Third_Order_Edges( cv::Mat img );
  void preprocessing( cv::Mat image );
  void convolve_img();
  int non_maximum_suppresion();

  void read_array_from_file(std::string filename, double *rd_data, int first_dim, int second_dim);
  void write_array_to_file(std::string filename, double *wr_data, int first_dim, int second_dim);

  std::vector<cv::Point2d> toed_locations;
  std::vector<double> toed_orientations;
  int Total_Num_Of_TOED;
  
};

#endif    // TOED_HPP