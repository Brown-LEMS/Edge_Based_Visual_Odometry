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

Eigen::Vector3d Utility::two_view_linear_triangulation(
  const Eigen::Vector3d gamma1, const Eigen::Vector3d gamma2,
  const Eigen::Matrix3d K1,     const Eigen::Matrix3d K2,
  const Eigen::Matrix3d Rel_R,  const Eigen::Vector3d Rel_T)
{
  Eigen::MatrixXd A(4, 4);
  Eigen::MatrixXd ATA(4, 4);
  Eigen::Vector4d GAMMA;
  Eigen::Vector3d pt3D;

  //> Convert points in pixels to points in meters
  Eigen::Vector3d gamma1_bar = K1.inverse() * gamma1;
  Eigen::Vector3d gamma2_bar = K2.inverse() * gamma2;

  //> The 3D point GAMMA is under the first camera coordinate,
  //  so R1 is an identity matrix and T1 is all zeros
  A(0,0) = 0.0; A(0,1) = -1.0; A(0,2) = gamma1_bar(1);  A(0,3) = 0.0;
  A(1,0) = 1.0; A(1,1) = 0.0;  A(1,2) = -gamma1_bar(0); A(1,3) = 0.0;

  double r1 = Rel_R(0,0), r2 = Rel_R(0,1), r3 = Rel_R(0,2), t1 = Rel_T(0);
  double r4 = Rel_R(1,0), r5 = Rel_R(1,1), r6 = Rel_R(1,2), t2 = Rel_T(1);
  double r7 = Rel_R(2,0), r8 = Rel_R(2,1), r9 = Rel_R(2,2), t3 = Rel_T(2);

  A(2, 0) = gamma2_bar(1) * r7 - r4;
  A(2, 1) = gamma2_bar(1) * r8 - r5; 
  A(2, 2) = gamma2_bar(1) * r9 - r6; 
  A(2, 3) = gamma2_bar(1) * t3 - t2;
  A(3, 0) = r1 - gamma2_bar(0) * r7; 
  A(3, 1) = r2 - gamma2_bar(0) * r8; 
  A(3, 2) = r3 - gamma2_bar(0) * r9; 
  A(3, 3) = t1 - gamma2_bar(0) * t3;

  //> Solving the homogeneous linear system and divide the first three rows with the last element
  ATA = A.transpose() * A;
  GAMMA = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col( ATA.rows() - 1 );
  GAMMA[0] /= GAMMA[3];
  GAMMA[1] /= GAMMA[3];
  GAMMA[2] /= GAMMA[3];
  
  //> Assign GAMMA to the returned point
  pt3D[0] = GAMMA[0];
  pt3D[1] = GAMMA[1];
  pt3D[2] = GAMMA[2];
  
  return pt3D;
}

Eigen::Vector3d Utility::multiview_linear_triangulation(
  const int N,
  const std::vector<Eigen::Vector2d> pts, 
  const std::vector<Eigen::Matrix3d> & Rs,
  const std::vector<Eigen::Vector3d> & Ts,
  const Eigen::Matrix3d K )
{
  Eigen::MatrixXd A(2*N, 4);
  Eigen::MatrixXd ATA(4, 4);
  Eigen::Vector4d GAMMA;
  Eigen::Vector3d pt3D; //> Returned variable

  //> Convert points in pixels to points in meters
  std::vector<Eigen::Vector2d> pts_meters;
  for (int p = 0; p < N; p++) {
      Eigen::Vector2d gamma;
      Eigen::Vector3d homo_p{pts[p](0), pts[p](1), 1.0};
      Eigen::Vector3d p_bar = K.inverse() * homo_p;
      gamma(0) = p_bar(0);
      gamma(1) = p_bar(1);
      pts_meters.push_back(gamma);
  }

  //> We are computing GAMMA under the first camera coordinate,
  //  so R1 is an identity matrix and T1 is all zeros
  A(0,0) = 0.0; A(0,1) = -1.0; A(0,2) = pts_meters[0](1); A(0,3) = 0.0;
  A(1,0) = 1.0; A(1,1) = 0.0; A(1,2) = -pts_meters[0](0); A(1,3) = 0.0;

  int row_cnter = 2;
  for (int p = 0; p < N-1; p++) {
      Eigen::Matrix3d Rp = Rs[p];
      Eigen::Vector3d Tp = Ts[p];
      Eigen::Vector2d mp = pts_meters[p+1];

      // std::cout << "Rp: " << Rp <<std::endl;
      
      double r1 = Rp(0,0), r2 = Rp(0,1), r3 = Rp(0,2), t1 = Tp(0);
      double r4 = Rp(1,0), r5 = Rp(1,1), r6 = Rp(1,2), t2 = Tp(1);
      double r7 = Rp(2,0), r8 = Rp(2,1), r9 = Rp(2,2), t3 = Tp(2);

      A(row_cnter,   0) = mp(1) * r7 - r4;
      A(row_cnter,   1) = mp(1) * r8 - r5; 
      A(row_cnter,   2) = mp(1) * r9 - r6; 
      A(row_cnter,   3) = mp(1) * t3 - t2;
      A(row_cnter+1, 0) = r1 - mp(0) * r7; 
      A(row_cnter+1, 1) = r2 - mp(0) * r8; 
      A(row_cnter+1, 2) = r3 - mp(0) * r9; 
      A(row_cnter+1, 3) = t1 - mp(0) * t3;
      row_cnter += 2;
  }

  //> Solving the homogeneous linear system and divide the first three rows with the last element
  ATA = A.transpose() * A;
  GAMMA = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col( ATA.rows() - 1 );
  GAMMA[0] /= GAMMA[3];
  GAMMA[1] /= GAMMA[3];
  GAMMA[2] /= GAMMA[3];
  
  //> Assign GAMMA to the returned point
  pt3D[0] = GAMMA[0];
  pt3D[1] = GAMMA[1];
  pt3D[2] = GAMMA[2];
  
  return pt3D;
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