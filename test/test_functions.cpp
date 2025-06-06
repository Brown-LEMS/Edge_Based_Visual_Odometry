
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "../include/utility.h"
#include "../include/Frame.h"
#include "../include/definitions.h"

class test_Data_Reader {
    
public:
    //> Allow this class to be accessed as a pointer
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //> Constructor
    test_Data_Reader( std::string );

    void read_3D_and_2D_point_correspondences();
    void read_gradient_rhos();
    double fx, fy, cx, cy;

    //> Data
    std::vector<Eigen::Vector3d> test_Curr_3D_Gamma;
    std::vector<Eigen::Vector3d> test_Prev_3D_Gamma;
    std::vector<Eigen::Vector3d> test_Curr_2D_gamma;
    std::vector<Eigen::Vector3d> test_Prev_2D_gamma;
    std::vector< std::pair<double, double>> test_Curr_gradient_Depth_at_Features;
    std::vector< std::pair<double, double>> test_Prev_gradient_Depth_at_Features;
private:
    std::string test_data_path;
};

test_Data_Reader::test_Data_Reader( std::string test_data_path ) : test_data_path(test_data_path) {}

void test_Data_Reader::read_3D_and_2D_point_correspondences( ) {
    
    //> Read Gammas and gammas
    std::string Curr_3D_Gammas_Path = test_data_path + "Curr_Gammas.txt";
    std::string Prev_3D_Gammas_Path = test_data_path + "Prev_Gammas.txt";
    std::string Curr_2D_gammas_Path = test_data_path + "Curr_gammas.txt";
    std::string Prev_2D_gammas_Path = test_data_path + "Prev_gammas.txt";

    std::fstream Curr_3D_Gammas_File;
    std::fstream Prev_3D_Gammas_File;
    std::fstream Curr_2D_gammas_File;
    std::fstream Prev_2D_gammas_File;

    Curr_3D_Gammas_File.open(Curr_3D_Gammas_Path, std::ios_base::in);
    Prev_3D_Gammas_File.open(Prev_3D_Gammas_Path, std::ios_base::in);
    Curr_2D_gammas_File.open(Curr_2D_gammas_Path, std::ios_base::in);
    Prev_2D_gammas_File.open(Prev_2D_gammas_Path, std::ios_base::in);

    double X, Y, Z;
    //> 3D points of current frame
    if (!Curr_3D_Gammas_File) {
        LOG_FILE_ERROR(Curr_3D_Gammas_Path);
        exit(1);
    }
    else {
        while (Curr_3D_Gammas_File >> X >> Y >> Z) {
            Eigen::Vector3d Gamma{ X, Y, Z };
            test_Curr_3D_Gamma.push_back( Gamma );
        }
    }
    Curr_3D_Gammas_File.close();

    //> 3D points of previous frame
    if (!Prev_3D_Gammas_File) {
        LOG_FILE_ERROR(Prev_3D_Gammas_Path);
        exit(1);
    }
    else {
        while (Prev_3D_Gammas_File >> X >> Y >> Z) {
            Eigen::Vector3d Gamma{ X, Y, Z };
            test_Prev_3D_Gamma.push_back( Gamma );
        }
    }
    Prev_3D_Gammas_File.close();

    //> 2D points of current frame
    if (!Curr_2D_gammas_File) {
        LOG_FILE_ERROR(Curr_2D_gammas_Path);
        exit(1);
    }
    else {
        while (Curr_2D_gammas_File >> X >> Y) {
            Eigen::Vector3d Gamma{ X, Y, 1.0 };
            test_Curr_2D_gamma.push_back( Gamma );
        }
    }
    Curr_2D_gammas_File.close();

    //> 2D points of previous frame
    if (!Prev_2D_gammas_File) {
        LOG_FILE_ERROR(Prev_2D_gammas_Path);
        exit(1);
    }
    else {
        while (Prev_2D_gammas_File >> X >> Y) {
            Eigen::Vector3d Gamma{ X, Y, 1.0 };
            test_Prev_2D_gamma.push_back( Gamma );
        }
    }
    Prev_2D_gammas_File.close();

    //> Check if the sizes are consistent
    assert(test_Curr_3D_Gamma.size() == test_Prev_3D_Gamma.size());
    assert(test_Curr_2D_gamma.size() == test_Prev_2D_gamma.size());
    assert(test_Curr_3D_Gamma.size() == test_Curr_2D_gamma.size());
}

void test_Data_Reader::read_gradient_rhos() {

    //> Read gradient depths
    std::string Curr_Grad_Rho_Path = test_data_path + "Curr_Grad_Rho.txt";
    std::string Prev_Grad_Rho_Path = test_data_path + "Prev_Grad_Rho.txt";

    std::fstream Curr_Grad_Rho_File;
    std::fstream Prev_Grad_Rho_File;

    Curr_Grad_Rho_File.open(Curr_Grad_Rho_Path, std::ios_base::in);
    Prev_Grad_Rho_File.open(Prev_Grad_Rho_Path, std::ios_base::in);
    
    //> gradient rho at xi and eta (grad_x and grad_y) of current frame
    double grad_x, grad_y;
    if (!Curr_Grad_Rho_File) {
        LOG_FILE_ERROR(Curr_Grad_Rho_Path);
        exit(1);
    }
    else {
        while (Curr_Grad_Rho_File >> grad_x >> grad_y) {
            test_Curr_gradient_Depth_at_Features.push_back( std::make_pair(grad_x, grad_y) );
        }
    }
    Curr_Grad_Rho_File.close();

    //> gradient rho at xi and eta (grad_x and grad_y) of previous frame
    if (!Prev_Grad_Rho_File) {
        LOG_FILE_ERROR(Prev_Grad_Rho_Path);
        exit(1);
    }
    else {
        while (Prev_Grad_Rho_File >> grad_x >> grad_y) {
            test_Prev_gradient_Depth_at_Features.push_back( std::make_pair(grad_x, grad_y) );
        }
    }
    Prev_Grad_Rho_File.close();

    //> check if the size is consistent
    assert(test_Curr_gradient_Depth_at_Features.size() == test_Prev_gradient_Depth_at_Features.size());
}

double get_GCC_dist( std::vector<Eigen::Vector3d> Prev_3D_Gammas, std::vector<Eigen::Vector3d> Curr_3D_Gammas, \
                     std::vector<Eigen::Vector3d> Prev_2D_gammas, std::vector<Eigen::Vector3d> Curr_2D_gammas, \
                     std::vector< std::pair<double, double>> Prev_gradient_Depth_at_Features, \
                     std::vector< std::pair<double, double>> Curr_gradient_Depth_at_Features, \
                     int anchor_index, int picked_index, Eigen::Matrix3d K, Eigen::Matrix3d invK ) {

    //> View 1 is previous frame, view 2 is current frame
    // std::cout << "Prev_3D_Gammas[ anchor_index ] = " << Prev_3D_Gammas[ anchor_index ] << std::endl;
    // std::cout << "Prev_3D_Gammas[ picked_index ] = " << Prev_3D_Gammas[ picked_index ] << std::endl;

    double phi_view1 = (Prev_3D_Gammas[ anchor_index ] - Prev_3D_Gammas[ picked_index ]).norm();
    double phi_view2 = (Curr_3D_Gammas[ anchor_index ] - Curr_3D_Gammas[ picked_index ]).norm();    //> ???
    Eigen::Vector3d gamma_view2   = invK * Curr_2D_gammas[ picked_index ]; //> Different!
    Eigen::Vector3d gamma_0_view2 = invK * Curr_2D_gammas[ anchor_index ]; //> Different!

    double rho_0 = (Curr_3D_Gammas[ anchor_index ])(2);
    double rho_p = (Curr_3D_Gammas[ picked_index ])(2);

    // std::cout << "phi_view1 = " << phi_view1 << std::endl;
    // std::cout << "phi_view2 = " << phi_view2 << std::endl;
    // std::cout << "gamma_view2 = " << gamma_view2 << std::endl;
    // std::cout << "gamma_0_view2 = " << gamma_0_view2 << std::endl;
    // std::cout << "rho_0 = " << rho_0 << std::endl;
    // std::cout << "rho_p = " << rho_p << std::endl;

    double gradient_phi_xi  = 2*(rho_p*(gamma_view2.norm()*gamma_view2.norm()) + rho_0*(gamma_view2.dot(gamma_0_view2))) * Curr_gradient_Depth_at_Features[ picked_index ].first \
                            + 2*rho_p*( (1.0/K(0,0))*(rho_p*gamma_view2(0) - rho_0*gamma_0_view2(0)) );
    double gradient_phi_eta = 2*(rho_p*(gamma_view2.norm()*gamma_view2.norm()) + rho_0*(gamma_view2.dot(gamma_0_view2))) * Curr_gradient_Depth_at_Features[ picked_index ].second \
                            + 2*rho_p*( (1.0/K(1,1))*(rho_p*gamma_view2(1) - rho_0*gamma_0_view2(1)) );

    // std::cout << "gradient_phi_xi = " << gradient_phi_xi << std::endl;
    // std::cout << "gradient_phi_eta = " << gradient_phi_eta << std::endl;

    double gradient_phi = sqrt(gradient_phi_xi*gradient_phi_xi + gradient_phi_eta*gradient_phi_eta);

    // std::cout << "gradient_phi = " << gradient_phi << std::endl;
    return fabs(phi_view1 - phi_view2) / gradient_phi;
}

void get_Relative_Pose_by_Three_Points_Alignment( 
    std::vector<Eigen::Vector3d> Prev_3D_Gammas, std::vector<Eigen::Vector3d> Curr_3D_Gammas, int Sample_Indices[3], \
    Eigen::Matrix3d &Estimated_Rel_Rot, Eigen::Vector3d &Estimated_Rel_Transl ) 
{
    
    std::vector<Eigen::Vector3d> Prev_Frame_Gammas;
    std::vector<Eigen::Vector3d> Curr_Frame_Gammas;
    Eigen::Matrix3d Prev_Frame_Shifted_Gammas;
    Eigen::Matrix3d Curr_Frame_Shifted_Gammas;
    Prev_Frame_Gammas.push_back( Prev_3D_Gammas[ Sample_Indices[0] ] );
    Prev_Frame_Gammas.push_back( Prev_3D_Gammas[ Sample_Indices[1] ] );
    Prev_Frame_Gammas.push_back( Prev_3D_Gammas[ Sample_Indices[2] ] );
    Curr_Frame_Gammas.push_back( Curr_3D_Gammas[ Sample_Indices[0] ] );
    Curr_Frame_Gammas.push_back( Curr_3D_Gammas[ Sample_Indices[1] ] );
    Curr_Frame_Gammas.push_back( Curr_3D_Gammas[ Sample_Indices[2] ] );
    Eigen::Vector3d Centroid_Prev = { (Prev_Frame_Gammas[0](0) + Prev_Frame_Gammas[1](0) + Prev_Frame_Gammas[2](0))/(double)(3), \
                                      (Prev_Frame_Gammas[0](1) + Prev_Frame_Gammas[1](1) + Prev_Frame_Gammas[2](1))/(double)(3), \
                                      (Prev_Frame_Gammas[0](2) + Prev_Frame_Gammas[1](2) + Prev_Frame_Gammas[2](2))/(double)(3)
                                    };
    Eigen::Vector3d Centroid_Curr = { (Curr_Frame_Gammas[0](0) + Curr_Frame_Gammas[1](0) + Curr_Frame_Gammas[2](0))/(double)(3), \
                                      (Curr_Frame_Gammas[0](1) + Curr_Frame_Gammas[1](1) + Curr_Frame_Gammas[2](1))/(double)(3), \
                                      (Curr_Frame_Gammas[0](2) + Curr_Frame_Gammas[1](2) + Curr_Frame_Gammas[2](2))/(double)(3)
                                    };
    //> Shift the 3D point Gammas by the centroid point
    for (int i = 0; i < 3; i++) {
        Prev_Frame_Shifted_Gammas.row(i) = Prev_Frame_Gammas[i] - Centroid_Prev;
        Curr_Frame_Shifted_Gammas.row(i) = Curr_Frame_Gammas[i] - Centroid_Curr;
    }

    //> Compute covariance matrix
    Eigen::Matrix3d Cov_Matrix = Prev_Frame_Shifted_Gammas.transpose() * Curr_Frame_Shifted_Gammas;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd( Cov_Matrix, Eigen::ComputeFullU | Eigen::ComputeFullV );

    Estimated_Rel_Rot = svd.matrixV() * svd.matrixU().transpose();

    if (Estimated_Rel_Rot.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V(0,2) *= -1.0;
        V(1,2) *= -1.0;
        V(2,2) *= -1.0;
        Estimated_Rel_Rot = V * svd.matrixU().transpose();
    }
    Estimated_Rel_Transl = Centroid_Curr - Estimated_Rel_Rot * Centroid_Prev;
}

int get_Hypothesis_Support_Reproject_from_3D_Points( std::vector<Eigen::Vector3d> Prev_3D_Gammas, std::vector<Eigen::Vector3d> Curr_2D_gammas, \
                                                     Eigen::Matrix3d R, Eigen::Vector3d T, Eigen::Matrix3d K ) {
    
    //> transform from the previous frame to the current frame and count the number of inliers
    int Num_Of_Inlier_Support = 0;
    for (size_t i = 0; i < Prev_3D_Gammas.size(); i++) {
        Eigen::Vector3d Transf_Gamma = R * Prev_3D_Gammas[i] + T;
        Transf_Gamma(0) /= Transf_Gamma(2);
        Transf_Gamma(1) /= Transf_Gamma(2);
        Eigen::Vector3d Projected_Point = {Transf_Gamma(0)*K(0,0) + K(0,2), Transf_Gamma(1)*K(1,1) + K(1,2), 1.0};
        
        //> Reprojection error
        double Reproj_Err = (Projected_Point - Curr_2D_gammas[i]).norm();

        //> Measure inlier support
        if ( Reproj_Err < REPROJ_ERROR_THRESH )
            Num_Of_Inlier_Support++;
    }
    return Num_Of_Inlier_Support;
}

double ComputeNCC(const cv::Mat patch_one, const cv::Mat patch_two){
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one  = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two  = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    cv::Mat norm_one = (patch_one - mean_one) / sqrt(sum_of_squared_one);
    cv::Mat norm_two = (patch_two - mean_two) / sqrt(sum_of_squared_two);
    return norm_one.dot(norm_two);
}

template< typename T >
void write_cvMat_to_File( std::string file_Path, cv::Mat Input_Mat ) {
    std::ofstream Test_File_Write_Stream;
    Test_File_Write_Stream.open(file_Path);
    if ( !Test_File_Write_Stream.is_open() ) LOG_FILE_ERROR(file_Path);
    for (int i = 0; i < Input_Mat.rows; i++) {
        for (int j = 0; j < Input_Mat.cols; j++) {
            Test_File_Write_Stream << Input_Mat.at<T>(i,j) << "\t";
        }
        Test_File_Write_Stream << "\n";
    }
    Test_File_Write_Stream.close();
}

cv::Point2d Epipolar_Shift( 
    cv::Point2d original_edge_location, double edge_orientation, \
    std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check )
{
    cv::Point2d corrected_edge;
    assert(epipolar_line_coeffs.size() == 3);
    double EL_coeff_A = epipolar_line_coeffs[0];
    double EL_coeff_B = epipolar_line_coeffs[1];
    double EL_coeff_C = epipolar_line_coeffs[2];
    double a1_line  = -epipolar_line_coeffs[0] / epipolar_line_coeffs[1];
    double b1_line  = -1;
    double c1_line  = -epipolar_line_coeffs[2] / epipolar_line_coeffs[1];
    
    //> Parameters of the line passing through the original edge along its direction (tangent) vector
    double a_edgeH2 = tan(edge_orientation);
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2*original_edge_location.x - original_edge_location.y); //−(a⋅x2−y2)

    //> Find the intersected point of the two lines
    corrected_edge.x = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    corrected_edge.y = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    
    //> Find (i) the displacement between the original edge and the corrected edge, and \
    //       (ii) the intersection angle between the epipolar line and the line passing through the original edge along its direction vector
    double epipolar_shift_displacement = cv::norm(corrected_edge - original_edge_location);
    double m_epipolar = -a1_line / b1_line; //> Slope of epipolar line
    double angle_diff_rad = abs(edge_orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180){
        angle_diff_deg -= 180;
    }

    //> check if the corrected edge passes the epoplar tengency test (intersection angle < 4 degrees and displacement < 6 pixels)
    b_pass_epipolar_tengency_check = (epipolar_shift_displacement < 6 && abs(angle_diff_deg - 0) > 4 && abs(angle_diff_deg - 180) > 4) ? (true) : (false);
    
    return corrected_edge;
}

int main(int argc, char **argv) {

#define TEST_OPENCV_OPERATION               (true)
#define TEST_NCC                            (true)
#define TEST_GAUSSIAN_DERIVATIVE_KERNEL     (false)
#define TEST_DEPTH_MAP                      (false)
#define TEST_GRADIENT_DEPTHS                (false)
#define TEST_SVD_EIGEN_MATRIX               (false)
#define TEST_RELATIVE_POSE                  (false)
#define TEST_GCC_CONSTRAINT                 (false)

    Utility utility_tool;

#if TEST_OPENCV_OPERATION
    //> [TEST] matrix operation using OpenCV's built-in function
    cv::Mat patch1 = (cv::Mat_<double>(7, 7) << 
        192, 195, 189, 183, 187, 184, 190,
        187, 185, 185, 186, 188, 188, 189,
        184, 183, 186, 187, 191, 190, 193,
        185, 184, 188, 188, 192, 190, 192,
        190, 188, 187, 190, 190, 192, 192,
        193, 191, 189, 192, 190, 192, 193,
        194, 193, 192, 194, 192, 193, 194);
    cv::Mat patch2 = (cv::Mat_<double>(7, 7) << 
        193, 190, 189, 189, 191, 191, 192,
        188, 188, 185, 182, 184, 185, 191,
        181, 181, 186, 185, 185, 186, 189,
        180, 187, 190, 185, 186, 191, 196,
        186, 187, 185, 186, 187, 187, 189,
        191, 188, 188, 190, 189, 191, 195,
        190, 187, 190, 192, 191, 192, 196);

    //> (i) element-wise multiplication
    cv::Mat patch_ele_mul = patch1.mul(patch2);
    std::cout << "element-wise multiplication of two patches: " << patch_ele_mul << std::endl;
    //> (ii) Calculate the dot product
    double patch_dot_prod = patch1.dot(patch2);
    std::cout << "dot product of two patches: " << patch_dot_prod << std::endl;
#endif
#if TEST_NCC
    double ncc = ComputeNCC(patch1, patch2);
    std::cout << "NCC of two patches: " << ncc << std::endl;
#endif
#if TEST_GAUSSIAN_DERIVATIVE_KERNEL
    //> [TEST] Gaussian derivative kernel
    cv::Mat Gx_2d, Gy_2d;
    Gx_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    Gy_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    utility_tool.get_dG_2D(Gx_2d, Gy_2d, 4*DEPTH_GRAD_GAUSSIAN_SIGMA, DEPTH_GRAD_GAUSSIAN_SIGMA); 
    std::cout << "Size of Gx_2d and Gy_2d: (" << Gx_2d.rows << ", " << Gx_2d.cols << ")" << std::endl;
    std::string write_Gx_2d_File_Name = std::string("../") + OUTPUT_WRITE_PATH + std::string("Test_Gx_2d.txt");
    write_cvMat_to_File<double>(write_Gx_2d_File_Name, Gx_2d);
#endif
#if TEST_DEPTH_MAP
    //> [TEST] Depth map
    cv::Mat depth_Map;
    std::string Current_Depth_Path = std::string("/home/chchien/datasets/TUM-RGBD/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png");
    depth_Map = cv::imread(Current_Depth_Path, cv::IMREAD_ANYDEPTH);
    if (depth_Map.data == nullptr) std::cerr << "ERROR: Cannot find depth map " << Current_Depth_Path << std::endl;
    //> Scale down by the factor of 5000 (according to the TUM-RGBD dataset)
    depth_Map.convertTo(depth_Map, CV_64F);
    depth_Map /= 5000.0;
    std::string write_Depth_Map_File_Name = std::string("../") + OUTPUT_WRITE_PATH + std::string("Test_Depth_Map.txt");
    write_cvMat_to_File<double>(write_Depth_Map_File_Name, depth_Map);
#endif
#if TEST_GRADIENT_DEPTHS
    //> [TEST] Gradient depths
    cv::Mat grad_Depth_xi_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    cv::Mat grad_Depth_eta_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    cv::filter2D( depth_Map, grad_Depth_xi_,  depth_Map.depth(), Gx_2d );
    cv::filter2D( depth_Map, grad_Depth_eta_, depth_Map.depth(), Gy_2d );
    grad_Depth_xi_  *= (-1);
    grad_Depth_eta_ *= (-1);
    std::string write_Gradient_Depth_Map_xi_File_Name = std::string("../") + OUTPUT_WRITE_PATH + std::string("Test_Gradient_Depth_xi.txt");
    write_cvMat_to_File<double>(write_Gradient_Depth_Map_xi_File_Name, grad_Depth_xi_);
    std::string write_Gradient_Depth_Map_eta_File_Name = std::string("../") + OUTPUT_WRITE_PATH + std::string("Test_Gradient_Depth_eta.txt");
    write_cvMat_to_File<double>(write_Gradient_Depth_Map_eta_File_Name, grad_Depth_eta_);
#endif

    //> [TEST] uniform random number generation
    // LOG_TEST("Uniform random number generator");
    // int rnd_sample = utility_tool.Uniform_Random_Number_Generator(0, 300);
    // std::cout << "random sample = " << rnd_sample << std::endl;
#if TEST_SVD_EIGEN_MATRIX
    //> [TEST] SVD of a Eigen::Matrix3d
    Eigen::Matrix3d Cov_Matrix;
    Cov_Matrix << 0.026752725823350, -0.009332331250622,  0.013799044199878, \
                 -0.014336743930935,  0.005001181722651, -0.007394886206797, \
                 -0.065749089150475,  0.022935691990062, -0.033913351233064;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd( Cov_Matrix, Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::Matrix3d Recover_Cov_Matrix = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    std::cout << "[TEST] Residual of SVD:" << std::endl << Cov_Matrix - Recover_Cov_Matrix << std::endl;
#endif
#if TEST_RELATIVE_POSE
    //> [TEST] relative rotation and translation
    Eigen::Vector3d curr_p1 = {-0.128738158692995,   0.188985781154671,   1.120163380809315};
    Eigen::Vector3d curr_p2 = { 0.406403095034157,   0.365184359243770,   0.890583764648437};
    Eigen::Vector3d curr_p3 = {-0.520822980142110,   0.054673591768922,   1.326050474628387};
    Eigen::Vector3d prev_p1 = { 0.321155165878628,   0.331102898915362,   0.780360092319548};
    Eigen::Vector3d prev_p2 = { 0.085091192248212,  -0.175343654707164,   1.203200000000000};
    Eigen::Vector3d prev_p3 = { 0.433489502281350,   0.169996750038386,   0.933266178009287};
    
    std::vector<Eigen::Vector3d> Prev_Frame_Gammas;
    std::vector<Eigen::Vector3d> Curr_Frame_Gammas;
    Eigen::Matrix3d Prev_Frame_Shifted_Gammas;
    Eigen::Matrix3d Curr_Frame_Shifted_Gammas;
    Prev_Frame_Gammas.push_back( prev_p1 );
    Prev_Frame_Gammas.push_back( prev_p2 );
    Prev_Frame_Gammas.push_back( prev_p3 );
    Curr_Frame_Gammas.push_back( curr_p1 );
    Curr_Frame_Gammas.push_back( curr_p2 );
    Curr_Frame_Gammas.push_back( curr_p3 );
    Eigen::Vector3d Centroid_Prev = { (Prev_Frame_Gammas[0](0) + Prev_Frame_Gammas[1](0) + Prev_Frame_Gammas[2](0))/(double)(3), \
                                      (Prev_Frame_Gammas[0](1) + Prev_Frame_Gammas[1](1) + Prev_Frame_Gammas[2](1))/(double)(3), \
                                      (Prev_Frame_Gammas[0](2) + Prev_Frame_Gammas[1](2) + Prev_Frame_Gammas[2](2))/(double)(3)
                                    };
    Eigen::Vector3d Centroid_Curr = { (Curr_Frame_Gammas[0](0) + Curr_Frame_Gammas[1](0) + Curr_Frame_Gammas[2](0))/(double)(3), \
                                      (Curr_Frame_Gammas[0](1) + Curr_Frame_Gammas[1](1) + Curr_Frame_Gammas[2](1))/(double)(3), \
                                      (Curr_Frame_Gammas[0](2) + Curr_Frame_Gammas[1](2) + Curr_Frame_Gammas[2](2))/(double)(3)
                                    };
    //> Shift the 3D point Gammas by the centroid point
    for (int i = 0; i < 3; i++) {
        Prev_Frame_Shifted_Gammas.row(i) = Prev_Frame_Gammas[i] - Centroid_Prev;
        Curr_Frame_Shifted_Gammas.row(i) = Curr_Frame_Gammas[i] - Centroid_Curr;
    }

    //> Compute covariance matrix
    Eigen::Matrix3d Cov_Matrix = Prev_Frame_Shifted_Gammas.transpose() * Curr_Frame_Shifted_Gammas;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd( Cov_Matrix, Eigen::ComputeFullU | Eigen::ComputeFullV );

    //> Compute relative rotation and translation
    Eigen::Matrix3d Rel_Rot = svd.matrixV() * svd.matrixU().transpose();
    if (Rel_Rot.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V(0,2) *= -1.0;
        V(1,2) *= -1.0;
        V(2,2) *= -1.0;
        Rel_Rot = V * svd.matrixU().transpose();
    }
    Eigen::Vector3d Rel_Transl = Centroid_Curr - Rel_Rot * Centroid_Prev;  
    std::cout << Rel_Rot << std::endl << std::endl;
    std::cout << Rel_Transl << std::endl;
#endif
#if TEST_GCC_CONSTRAINT
    //> [TEST] GDC constraint computation
    std::string test_data_path = "../test/test_data/";
    test_Data_Reader data_reader(test_data_path);
    data_reader.read_3D_and_2D_point_correspondences();
    data_reader.read_gradient_rhos();

    int Sample_Indices[3] = {81, 135, 25};
    double fx = 517.3000;
    double fy = 516.5000;
    double cx = 318.6000;
    double cy = 255.3000;
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0,0) = fx;
    K(0,2) = cx;
    K(1,1) = fy;
    K(1,2) = cy;
    Eigen::Matrix3d invK = K.inverse();
    double gcc_12 = get_GCC_dist( data_reader.test_Prev_3D_Gamma, data_reader.test_Curr_3D_Gamma, \
                                  data_reader.test_Prev_2D_gamma, data_reader.test_Curr_2D_gamma, \
                                  data_reader.test_Prev_gradient_Depth_at_Features, \
                                  data_reader.test_Curr_gradient_Depth_at_Features, \
                                  Sample_Indices[0], Sample_Indices[1], K, invK );
    std::cout << "gcc_12 = " << gcc_12 << std::endl;
    double gcc_21 = get_GCC_dist( data_reader.test_Curr_3D_Gamma, data_reader.test_Prev_3D_Gamma, \
                                  data_reader.test_Curr_2D_gamma, data_reader.test_Prev_2D_gamma, \
                                  data_reader.test_Curr_gradient_Depth_at_Features, \
                                  data_reader.test_Prev_gradient_Depth_at_Features, \
                                  Sample_Indices[0], Sample_Indices[1], K, invK );
    std::cout << "gcc_21 = " << gcc_21 << std::endl;
    double gcc_13 = get_GCC_dist( data_reader.test_Prev_3D_Gamma, data_reader.test_Curr_3D_Gamma, \
                                  data_reader.test_Prev_2D_gamma, data_reader.test_Curr_2D_gamma, \
                                  data_reader.test_Prev_gradient_Depth_at_Features, \
                                  data_reader.test_Curr_gradient_Depth_at_Features, \
                                  Sample_Indices[0], Sample_Indices[2], K, invK );
    std::cout << "gcc_13 = " << gcc_13 << std::endl;
    double gcc_31 = get_GCC_dist( data_reader.test_Curr_3D_Gamma, data_reader.test_Prev_3D_Gamma, \
                                  data_reader.test_Curr_2D_gamma, data_reader.test_Prev_2D_gamma, \
                                  data_reader.test_Curr_gradient_Depth_at_Features, \
                                  data_reader.test_Prev_gradient_Depth_at_Features, \
                                  Sample_Indices[0], Sample_Indices[2], K, invK );
    std::cout << "gcc_31 = " << gcc_31 << std::endl;

    Eigen::Matrix3d Estimated_Rel_Rot;
    Eigen::Vector3d Estimated_Rel_Transl;
    get_Relative_Pose_by_Three_Points_Alignment( data_reader.test_Prev_3D_Gamma, data_reader.test_Curr_3D_Gamma, Sample_Indices, Estimated_Rel_Rot, Estimated_Rel_Transl );

    std::cout << "Estimated relative rotation:" << std::endl;
    std::cout << Estimated_Rel_Rot << std::endl;
    std::cout << "Estimated relative translation:" << std::endl;
    std::cout << Estimated_Rel_Transl << std::endl;

    int Num_Of_Inliers = get_Hypothesis_Support_Reproject_from_3D_Points( data_reader.test_Prev_3D_Gamma, data_reader.test_Curr_2D_gamma, \
                                                                          Estimated_Rel_Rot, Estimated_Rel_Transl, K );
    std::cout << "Number of inliers = " << Num_Of_Inliers << std::endl;
#endif
    // std::cout << "- Current frame 3D points:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Curr_3D_Gamma[i])(0) << "\t" << (data_reader.test_Curr_3D_Gamma[i])(1) << "\t" << (data_reader.test_Curr_3D_Gamma[i])(2) << std::endl;
    // std::cout << "- Previous frame 3D points:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Prev_3D_Gamma[i])(0) << "\t" << (data_reader.test_Prev_3D_Gamma[i])(1) << "\t" << (data_reader.test_Prev_3D_Gamma[i])(2) << std::endl;
    // std::cout << "- Current frame 2D points:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Curr_2D_gamma[i])(0) << "\t" << (data_reader.test_Curr_2D_gamma[i])(1) << "\t" << (data_reader.test_Curr_2D_gamma[i])(2) << std::endl;
    // std::cout << "- Previous frame 2D points:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Prev_2D_gamma[i])(0) << "\t" << (data_reader.test_Prev_2D_gamma[i])(1) << "\t" << (data_reader.test_Prev_2D_gamma[i])(2) << std::endl;
    // std::cout << "- Current frame gradient depths:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Curr_gradient_Depth_at_Features[i]).first << ", " << (data_reader.test_Curr_gradient_Depth_at_Features[i]).second << std::endl;
    // std::cout << "- Previous frame gradient depths:" << std::endl;
    // for (int i = 0; i < 10; i++) std::cout << (data_reader.test_Prev_gradient_Depth_at_Features[i]).first << ", " << (data_reader.test_Prev_gradient_Depth_at_Features[i]).second << std::endl;
    
    
    return 0;
}