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
#include "toed/cpu_toed.hpp"

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

//> Define and initialize static members (basis vectors in 3D space)
Eigen::Vector3d Utility::e1 = Eigen::Vector3d::UnitX();
Eigen::Vector3d Utility::e2 = Eigen::Vector3d::UnitY();
Eigen::Vector3d Utility::e3 = Eigen::Vector3d::UnitZ();

//> Normal distance from an edge to the corresponding epipolar line
double Utility::getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y)
{
  double a1_line = Epip_Line_Coeffs(0);
  double b1_line = Epip_Line_Coeffs(1);
  double c1_line = Epip_Line_Coeffs(2);
  epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line) / (pow(a1_line, 2) + pow(b1_line, 2));
  epiline_y = edge(1) - b1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line) / (pow(a1_line, 2) + pow(b1_line, 2));
  return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

double Utility::getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &epiline_x, double &epiline_y)
{
  Eigen::Vector3d edge(edges(index, 0), edges(index, 1), 1.0);
  return getNormalDistance2EpipolarLine(Epip_Line_Coeffs, edge, epiline_x, epiline_y);
}

//> Tangential distance from an edge to the corresponding epipolar line
double Utility::getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection)
{
  double a_edgeH2 = tan(edge(2)); // tan(theta2)
  double b_edgeH2 = -1;
  double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1)); // −(a⋅x2−y2)
  double a1_line = Epip_Line_Coeffs(0);
  double b1_line = Epip_Line_Coeffs(1);
  double c1_line = Epip_Line_Coeffs(2);
  x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
  y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
  return sqrt((x_intersection - edge(0)) * (x_intersection - edge(0)) + (y_intersection - edge(1)) * (y_intersection - edge(1)));
}

double Utility::getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &x_intersection, double &y_intersection)
{
  Eigen::Vector3d edge(edges(index, 0), edges(index, 1), edges(index, 2));
  return getTangentialDistance2EpipolarLine(Epip_Line_Coeffs, edge, x_intersection, y_intersection);
}

std::pair<cv::Point2d, cv::Point2d> Utility::get_Orthogonal_Shifted_Points(const Edge edgel)
{
  double shifted_x1 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel.orientation));
  double shifted_y1 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel.orientation));
  double shifted_x2 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel.orientation));
  double shifted_y2 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel.orientation));

  cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
  cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

  return {shifted_point_plus, shifted_point_minus};
}

Eigen::Vector3d Utility::backproject_2D_point_to_3D_point_using_rays( const Eigen::Matrix3d rel_R, const Eigen::Vector3d rel_T, const Eigen::Vector3d ray1, const Eigen::Vector3d ray2 )
{
    //> Analytically find the depth
    double numerator = e1.dot(rel_T) - (e3.dot(rel_T)) * e1.dot(ray2);
    double denominator = e3.dot(rel_R * ray1) * (e1.dot(ray2)) - e1.dot(rel_R * ray1);
    double rho1 = numerator / denominator;
    return rho1 * ray1;
}

Eigen::Vector3d Utility::reconstruct_3D_Tangent( const Eigen::Matrix3d rel_R, Eigen::Vector3d gamma1, Eigen::Vector3d gamma2, Eigen::Vector3d tangent1, Eigen::Vector3d tangent2 )
{
  // Eigen::Vector3d T_3D = -(gamma2.dot(t2.cross(rel_R * t1))) * gamma1 + (gamma2.dot(t2.cross(rel_R * gamma1))) * t1;
  Eigen::Vector3d n1 = tangent1.cross(gamma1);
  Eigen::Vector3d n2 = rel_R.transpose() * (tangent2.cross(gamma2));
  Eigen::Vector3d T_3D = n1.cross(n2);
  T_3D.normalize();
  return T_3D;
}

Eigen::Vector3d Utility::project_3D_Tangent_to_2D_Tangent( const Eigen::Vector3d Tangent_3D, Eigen::Vector3d gamma )
{
  Eigen::Vector3d projected_tangent = Tangent_3D - Tangent_3D.z() * gamma;
  projected_tangent.normalize();
  return projected_tangent;
}

Camera_Pose Utility::get_Relative_Pose( const Camera_Pose &source_pose, const Camera_Pose &target_pose )
{
  Eigen::Matrix3d rel_R = target_pose.R * (source_pose.R).transpose();
  Eigen::Vector3d rel_T = -rel_R * source_pose.t + target_pose.t;
  return Camera_Pose(rel_R, rel_T);
}

std::pair<cv::Point2d, cv::Point2d> Utility::get_Orthogonal_Shifted_Points(const Edge edgel, double shift_magnitude)
{
  double shifted_x1 = edgel.location.x + shift_magnitude * (std::sin(edgel.orientation));
  double shifted_y1 = edgel.location.y + shift_magnitude * (-std::cos(edgel.orientation));
  double shifted_x2 = edgel.location.x + shift_magnitude * (-std::sin(edgel.orientation));
  double shifted_y2 = edgel.location.y + shift_magnitude * (std::cos(edgel.orientation));

  cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
  cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

  return {shifted_point_plus, shifted_point_minus};
}

void Utility::get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta,
                                         cv::Mat &patch_coord_x, cv::Mat &patch_coord_y,
                                         cv::Mat &patch_val, const cv::Mat img)
{
  const int half_patch_size = floor(PATCH_SIZE / 2);
  for (int i = -half_patch_size; i <= half_patch_size; i++)
  {
    for (int j = -half_patch_size; j <= half_patch_size; j++)
    {
      //> get the rotated coordinate
      cv::Point2d rotated_point(cos(theta) * (i)-sin(theta) * (j) + shifted_point.x, sin(theta) * (i) + cos(theta) * (j) + shifted_point.y);
      patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
      patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

      //> get the image intensity of the rotated coordinate
      // double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
      double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
      patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
    }
  }
}

double Utility::get_patch_similarity(const cv::Mat patch_one, const cv::Mat patch_two)
{
  double mean_one = (cv::mean(patch_one))[0];
  double mean_two = (cv::mean(patch_two))[0];
  double sum_of_squared_one = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
  double sum_of_squared_two = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

  const double epsilon = 1e-10;
  if (sum_of_squared_one < epsilon || sum_of_squared_two < epsilon)
    return -1.0;

  double denom_one = sqrt(sum_of_squared_one);
  double denom_two = sqrt(sum_of_squared_two);

  cv::Mat norm_one = (patch_one - mean_one) / denom_one;
  cv::Mat norm_two = (patch_two - mean_two) / denom_two;
  return norm_one.dot(norm_two);
}

std::pair<cv::Mat, cv::Mat> Utility::get_edge_patches(const Edge edge, const cv::Mat img, bool b_debug)
{
  std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points(edge);
  cv::Mat patch_val_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  cv::Mat patch_val_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  cv::Mat patch_coord_x_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  cv::Mat patch_coord_y_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
  get_patch_on_one_edge_side(shifted_points.first, edge.orientation, patch_coord_x_plus, patch_coord_y_plus, patch_val_plus, img);
  get_patch_on_one_edge_side(shifted_points.second, edge.orientation, patch_coord_x_minus, patch_coord_y_minus, patch_val_minus, img);

  if (b_debug)
  {
    std::cout << "Edge location: " << edge.location << std::endl;
    std::cout << "Edge orientation: " << edge.orientation << std::endl;
    std::cout << "Patch (+) coordinates: " << std::endl;
    std::cout << patch_coord_x_plus << std::endl;
    std::cout << patch_coord_y_plus << std::endl;
    std::cout << "Patch (-) coordinates: " << std::endl;
    std::cout << patch_coord_x_minus << std::endl;
    std::cout << patch_coord_y_minus << std::endl;
  }

  if (patch_val_plus.type() != CV_32F)
    patch_val_plus.convertTo(patch_val_plus, CV_32F);
  if (patch_val_minus.type() != CV_32F)
    patch_val_minus.convertTo(patch_val_minus, CV_32F);

  return {patch_val_plus, patch_val_minus};
}

Eigen::Vector3d Utility::two_view_linear_triangulation(
    const Eigen::Vector3d gamma1, const Eigen::Vector3d gamma2,
    const Eigen::Matrix3d K1, const Eigen::Matrix3d K2,
    const Eigen::Matrix3d Rel_R, const Eigen::Vector3d Rel_T)
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
  A(0, 0) = 0.0;
  A(0, 1) = -1.0;
  A(0, 2) = gamma1_bar(1);
  A(0, 3) = 0.0;
  A(1, 0) = 1.0;
  A(1, 1) = 0.0;
  A(1, 2) = -gamma1_bar(0);
  A(1, 3) = 0.0;

  double r1 = Rel_R(0, 0), r2 = Rel_R(0, 1), r3 = Rel_R(0, 2), t1 = Rel_T(0);
  double r4 = Rel_R(1, 0), r5 = Rel_R(1, 1), r6 = Rel_R(1, 2), t2 = Rel_T(1);
  double r7 = Rel_R(2, 0), r8 = Rel_R(2, 1), r9 = Rel_R(2, 2), t3 = Rel_T(2);

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
  GAMMA = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col(ATA.rows() - 1);
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
    const std::vector<Eigen::Matrix3d> &Rs,
    const std::vector<Eigen::Vector3d> &Ts,
    const Eigen::Matrix3d K)
{
  Eigen::MatrixXd A(2 * N, 4);
  Eigen::MatrixXd ATA(4, 4);
  Eigen::Vector4d GAMMA;
  Eigen::Vector3d pt3D; //> Returned variable

  //> Convert points in pixels to points in meters
  std::vector<Eigen::Vector2d> pts_meters;
  for (int p = 0; p < N; p++)
  {
    Eigen::Vector2d gamma;
    Eigen::Vector3d homo_p{pts[p](0), pts[p](1), 1.0};
    Eigen::Vector3d p_bar = K.inverse() * homo_p;
    gamma(0) = p_bar(0);
    gamma(1) = p_bar(1);
    pts_meters.push_back(gamma);
  }

  //> We are computing GAMMA under the first camera coordinate,
  //  so R1 is an identity matrix and T1 is all zeros
  A(0, 0) = 0.0;
  A(0, 1) = -1.0;
  A(0, 2) = pts_meters[0](1);
  A(0, 3) = 0.0;
  A(1, 0) = 1.0;
  A(1, 1) = 0.0;
  A(1, 2) = -pts_meters[0](0);
  A(1, 3) = 0.0;

  int row_cnter = 2;
  for (int p = 0; p < N - 1; p++)
  {
    Eigen::Matrix3d Rp = Rs[p];
    Eigen::Vector3d Tp = Ts[p];
    Eigen::Vector2d mp = pts_meters[p + 1];

    // std::cout << "Rp: " << Rp <<std::endl;

    double r1 = Rp(0, 0), r2 = Rp(0, 1), r3 = Rp(0, 2), t1 = Tp(0);
    double r4 = Rp(1, 0), r5 = Rp(1, 1), r6 = Rp(1, 2), t2 = Tp(1);
    double r7 = Rp(2, 0), r8 = Rp(2, 1), r9 = Rp(2, 2), t3 = Tp(2);

    A(row_cnter, 0) = mp(1) * r7 - r4;
    A(row_cnter, 1) = mp(1) * r8 - r5;
    A(row_cnter, 2) = mp(1) * r9 - r6;
    A(row_cnter, 3) = mp(1) * t3 - t2;
    A(row_cnter + 1, 0) = r1 - mp(0) * r7;
    A(row_cnter + 1, 1) = r2 - mp(0) * r8;
    A(row_cnter + 1, 2) = r3 - mp(0) * r9;
    A(row_cnter + 1, 3) = t1 - mp(0) * t3;
    row_cnter += 2;
  }

  //> Solving the homogeneous linear system and divide the first three rows with the last element
  ATA = A.transpose() * A;
  GAMMA = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col(ATA.rows() - 1);
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