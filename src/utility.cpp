#ifndef UTILITY_CPP
#define UTILITY_CPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "Dataset.h"
#include "definitions.h"

// =======================================================================================================
// utility Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Chien  23-01-21    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

Utility::Utility() {}

//> Normal distance from an edge to the corresponding epipolar line
double Utility::getNormalDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y ) 
{
  double a1_line = Epip_Line_Coeffs(0);
  double b1_line = Epip_Line_Coeffs(1);
  double c1_line = Epip_Line_Coeffs(2);
  epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line *edge(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
  epiline_y = edge(1) - b1_line* (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
  return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

double Utility::getNormalDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &epiline_x, double &epiline_y ) 
{
  Eigen::Vector3d edge(edges(index, 0), edges(index, 1), 1.0);
  return getNormalDistance2EpipolarLine( Epip_Line_Coeffs, edge, epiline_x, epiline_y );
}

//> Tangential distance from an edge to the corresponding epipolar line
double Utility::getTangentialDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection ) {
  double a_edgeH2 = tan(edge(2)); //tan(theta2)
  double b_edgeH2 = -1;
  double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1)); //−(a⋅x2−y2)
  double a1_line = Epip_Line_Coeffs(0);
  double b1_line = Epip_Line_Coeffs(1);
  double c1_line = Epip_Line_Coeffs(2);
  x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
  y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
  return sqrt((x_intersection - edge(0))*(x_intersection - edge(0))+(y_intersection - edge(1))*(y_intersection - edge(1)));
}

double Utility::getTangentialDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &x_intersection, double &y_intersection ) {
  Eigen::Vector3d edge(edges(index, 0), edges(index, 1), edges(index, 2));
  return getTangentialDistance2EpipolarLine( Epip_Line_Coeffs, edge, x_intersection, y_intersection );
}

//> Display images and features via OpenCV
void Utility::Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2,
                                              std::vector<cv::KeyPoint> KeyPoint1, std::vector<cv::KeyPoint> KeyPoint2,
                                              std::vector<cv::DMatch> Good_Matches)
{
  //> Credit: matcher_simple.cpp from the official OpenCV
  cv::namedWindow("matches", 1);
  cv::Mat img_matches;
  cv::drawMatches(Img1, KeyPoint1, Img2, KeyPoint2, Good_Matches, img_matches);
  cv::imshow("matches", img_matches);
  cv::waitKey(0);
}

std::string Utility::cvMat_Type(int type)
{
  //> Credit: https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv

  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth)
  {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

// template<typename T>
// T Utility::Uniform_Random_Number_Generator(T range_from, T range_to) {

//   std::random_device                  rand_dev;
//   std::mt19937                        generator(rand_dev());
//   std::uniform_int_distribution<T>    distr(range_from, range_to);
//   return distr(generator);
// }

#endif