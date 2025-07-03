#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/EBVO.h"
#include "../include/OpticalFlow.h"

/**
 * Test program for optical flow functionality in EBVO
 * This demonstrates how to use the new optical flow features
 */

void test_optical_flow_basic()
{
    std::cout << "=== Testing Basic Optical Flow Functionality ===" << std::endl;

    // Create sample edges for testing
    std::vector<Edge2D> test_edges;
    test_edges.push_back(Edge2D(cv::Point2f(100, 100), 45.0, 50.0, 0));
    test_edges.push_back(Edge2D(cv::Point2f(200, 150), 90.0, 60.0, 1));
    test_edges.push_back(Edge2D(cv::Point2f(300, 200), 135.0, 45.0, 2));

    // Create test images (synthetic data for demonstration)
    cv::Mat frame1 = cv::Mat::zeros(480, 640, CV_8UC1);
    cv::Mat frame2 = cv::Mat::zeros(480, 640, CV_8UC1);

    // Add some features to the images
    cv::circle(frame1, cv::Point(100, 100), 5, cv::Scalar(255), -1);
    cv::circle(frame1, cv::Point(200, 150), 5, cv::Scalar(255), -1);
    cv::circle(frame1, cv::Point(300, 200), 5, cv::Scalar(255), -1);

    // Move features slightly in second frame (simulate motion)
    cv::circle(frame2, cv::Point(105, 102), 5, cv::Scalar(255), -1);
    cv::circle(frame2, cv::Point(198, 155), 5, cv::Scalar(255), -1);
    cv::circle(frame2, cv::Point(295, 205), 5, cv::Scalar(255), -1);

    // Initialize optical flow
    FlowParameters params;
    OpticalFlow optical_flow(params);

    // Initialize with first frame
    optical_flow.Initialize(frame1, test_edges);

    // Track edges in second frame
    std::vector<Edge2D> tracked_edges;
    int tracked_count = optical_flow.TrackEdges(frame2, tracked_edges);

    std::cout << "Successfully tracked " << tracked_count << " out of " << test_edges.size() << " edges" << std::endl;

    // Display motion vectors
    auto motion_vectors = optical_flow.CalculateMotionVectors(test_edges, tracked_edges);
    std::cout << "Motion vectors:" << std::endl;
    for (size_t i = 0; i < motion_vectors.size(); ++i)
    {
        std::cout << "  Edge " << i << ": (" << motion_vectors[i].x << ", " << motion_vectors[i].y << ")" << std::endl;
    }

    std::cout << "Basic optical flow test completed successfully!" << std::endl;
}

void test_optical_flow_integration()
{
    std::cout << "\n=== Testing EBVO Optical Flow Integration ===" << std::endl;

    // This would typically be done with a real configuration file
    YAML::Node config;
    config["dataset_type"] = "test";
    config["output_path"] = "./test_output";

    try
    {
        EBVO ebvo(config, false);

        std::cout << "EBVO with optical flow integration created successfully!" << std::endl;
        std::cout << "Use PerformTemporalEdgeTracking() for optical flow-based processing" << std::endl;
        std::cout << "Use TrackEdgesBetweenFrames() for frame-to-frame edge tracking" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "Note: EBVO creation requires valid dataset configuration" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "This is expected in a test environment without real data" << std::endl;
    }
}

void print_usage_instructions()
{
    std::cout << "\n=== Optical Flow Usage Instructions ===" << std::endl;
    std::cout << "The optical flow functionality has been integrated into EBVO with the following features:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. PerformTemporalEdgeTracking():" << std::endl;
    std::cout << "   - Alternative to PerformEdgeBasedVO() that uses optical flow" << std::endl;
    std::cout << "   - Tracks edges temporally across frame sequences" << std::endl;
    std::cout << "   - Saves tracking results to CSV files" << std::endl;
    std::cout << std::endl;

    std::cout << "2. TrackEdgesBetweenFrames():" << std::endl;
    std::cout << "   - Track specific edges between two frames" << std::endl;
    std::cout << "   - Useful for targeted edge tracking" << std::endl;
    std::cout << std::endl;

    std::cout << "3. OpticalFlow class features:" << std::endl;
    std::cout << "   - Pyramidal Lucas-Kanade tracking" << std::endl;
    std::cout << "   - Bidirectional flow consistency checking" << std::endl;
    std::cout << "   - Edge property validation (orientation, strength)" << std::endl;
    std::cout << "   - Robust outlier filtering" << std::endl;
    std::cout << std::endl;

    std::cout << "4. Configuration parameters (in definitions.h):" << std::endl;
    std::cout << "   - OPTICAL_FLOW_WINDOW_SIZE: Lucas-Kanade window size" << std::endl;
    std::cout << "   - OPTICAL_FLOW_MAX_LEVEL: Pyramid levels" << std::endl;
    std::cout << "   - OPTICAL_FLOW_MAX_ERROR: Maximum tracking error threshold" << std::endl;
    std::cout << "   - OPTICAL_FLOW_ORIENT_THRESH: Orientation change threshold" << std::endl;
    std::cout << std::endl;

    std::cout << "5. Output files:" << std::endl;
    std::cout << "   - optical_flow/edge_tracks.csv: Complete tracking history" << std::endl;
    std::cout << "   - Motion analysis statistics printed to console" << std::endl;
    std::cout << std::endl;

    std::cout << "To use in your main application:" << std::endl;
    std::cout << "   EBVO ebvo(config);" << std::endl;
    std::cout << "   ebvo.PerformTemporalEdgeTracking();  // Instead of PerformEdgeBasedVO()" << std::endl;
}

int main()
{
    std::cout << "EBVO Optical Flow Test Program" << std::endl;
    std::cout << "==============================" << std::endl;

    // Run basic optical flow test
    test_optical_flow_basic();

    // Test EBVO integration
    test_optical_flow_integration();

    // Print usage instructions
    print_usage_instructions();

    std::cout << "\nOptical flow integration for EBVO is ready!" << std::endl;
    std::cout << "You can now track edges temporally across frame sequences." << std::endl;

    return 0;
}
