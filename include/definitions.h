//> Macro definitions
// USE_GLOGS is now defined by CMake based on glog/gflags availability
#ifndef USE_GLOGS
#define USE_GLOGS (false)
#endif

#define USE_CPP17 (true)

//> OpenMP settings
#define ACTIVATE_OPENMP_SUPPORT (true)
#define USE_DEFINED_NUM_OF_CORES (false)
#if USE_DEFINED_NUM_OF_CORES
#define USE_NUM_CORES_FOR_OMP (8)
#endif

//> Stereo edge matching settings
#define EPIPOLAR_LINE_DIST_THRESH (0.5)  //> in pixels
#define EPIP_TENGENCY_ORIENT_THRESH (12) //> in degrees
#define EPIP_TENGENCY_PROXIM_THRESH (4)  //> in pixels
#define MAX_DISPARITY (25)
#define EDGE_CLUSTER_THRESH (0.3) //> in pixels
#define ORTHOGONAL_SHIFT_MAG (5)  //> in pixels
#define PATCH_SIZE (7)            //> in pixels
#define NCC_THRESH (0.6)

#define EPIP_TANGENCY_DISPL_THRESH (3) //> in pixels
#define LOCATION_PERTURBATION (0.4)    //> in pixels
#define ORIENT_PERTURBATION (0.174533) //> in radians. 0.174533 is 10 degrees
#define CLUSTER_DIST_THRESH (1)        //> τc, in pixels
#define CLUSTER_ORIENT_THRESH (20.0)   //> in degrees
#define MAX_CLUSTER_SIZE (10)          //> max number of edges per cluster
#define CLUSTER_ORIENT_GAUSS_SIGMA (2.0)
#define BNB_SIFT (0.4)
#define BNB_NCC (0.9)
#define HUBER_DELTA (1.0) //> Huber threshold
#define LOWES_RATIO (0.8) //> Suggested in Lowe's paper

#define BIDIRECTIONAL_FILTERING (false)

#define SIFT_THRESHOLD (500.0)
//> precision-recall experiments
#define DIST_TO_GT_THRESH (1.0) //> in pixels

#define GRID_SIZE (15) //> Size of the spatial grid cells in pixels

#define MEASURE_TIMINGS (false)

//> Generic definitions
#define RANSAC_NUM_OF_ITERATIONS (500)
#define EPSILON (1e-12)
#define REPROJ_ERROR_THRESH (2) //> in pixels

//> Verbose
#define DATASET_LOAD_VERBOSE (false)
#define STEREO_EDGE_MATCH_EVAL_VERBOSE (false)

//> Output for visualization
#define RECORD_FILTER_DISTRIBUTIONS (false)

//> DEBUGGING PURPOSE
#define SHOW_YAML_FILE_DATA (false)
#define DEBUG_FALSE_NEGATIVES (false)
#define DEBUG_COLLECT_NCC_AND_ERR (false)
#define DEBUG_EDGE_MATCHES_BETWEEN_LEFT_IMGS (false)
//> ----------------------------------------------
#define WRITE_FEATURES_TO_FILE (false)
#define WRITE_CORRESPONDENCES_TO_FILE (false)
#define OPENCV_DISPLAY_FEATURES (false)
#define OPENCV_DISPLAY_CORRESPONDENCES (false)
//> ----------------------------------------------

//> Third-Order Edge Detection Parameters
#define TOED_KERNEL_SIZE (17)
#define TOED_SIGMA (2)

//> SIFT parameters
#define SIFT_NFEATURES (0)
#define SIFT_NOCTAVE_LAYERS (4)
#define SIFT_CONTRAST_THRESHOLD (0.04)
#define SIFT_EDGE_THRESHOLD (10)
#define SIFT_GAUSSIAN_SIGMA (1.6)
#define K_IN_KNN_MATCHING (2)

//> Print outs
#define LOG_INFO(info_msg) printf("\033[1;32m[INFO] %s\033[0m\n", std::string(info_msg).c_str());
#define LOG_STATUS(status_) printf("\033[1;35m[STATUS] %s\033[0m\n", std::string(status_).c_str());
#define LOG_ERROR(err_msg) printf("\033[1;31m[ERROR] %s\033[0m\n", std::string(err_msg).c_str());
#define LOG_TEST(test_msg) printf("\033[1;30m[TEST] %s\033[0m\n", std::string(test_msg).c_str());
#define LOG_FILE_ERROR(err_msg) printf("\033[1;31m[ERROR] File %s not found!\033[0m\n", std::string(err_msg).c_str());
#define LOG_TRACE(msg) std::cout << "\033[1;30m[TRACE] " << __func__ << ":" << __LINE__ << " " << msg << "\033[0m" << std::endl;
#define LOG_PRINT_HELP_MESSAGE printf("Usage: ./main_VO [flag] [argument]\n\n"                 \
                                      "options:\n"                                             \
                                      "  -h, --help         show this help message and exit\n" \
                                      "  -c, --config_file  path to the the configuration file\n");
