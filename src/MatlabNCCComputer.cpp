#include "MatlabNCCComputer.h"

#ifdef USE_MATLAB_NCC

#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include <iostream>
#include <filesystem>

MatlabNCCComputer::MatlabNCCComputer() : initialized(false)
{
    factory = std::make_unique<matlab::data::ArrayFactory>();
}

MatlabNCCComputer::~MatlabNCCComputer()
{
    if (matlabPtr)
    {
        matlabPtr->eval(u"clear all;");
    }
}

bool MatlabNCCComputer::initialize()
{
    if (initialized)
    {
        return true;
    }

    try
    {
        std::cout << "Starting MATLAB engine..." << std::endl;
        matlabPtr = matlab::engine::startMATLAB();

        // Get the current directory and add it to MATLAB path
        std::string current_dir = std::filesystem::current_path().string();
        std::string src_dir = current_dir + "/src";

        std::u16string matlab_cmd = u"addpath('" +
                                    std::u16string(src_dir.begin(), src_dir.end()) + u"');";
        matlabPtr->eval(matlab_cmd);

        // Test if the function exists
        matlabPtr->eval(u"if ~exist('compute_ncc', 'file'), error('compute_ncc.m not found'); end");

        initialized = true;
        std::cout << "MATLAB engine initialized successfully!" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize MATLAB engine: " << e.what() << std::endl;
        initialized = false;
        return false;
    }
}

double MatlabNCCComputer::computeNCC(const cv::Mat &patch1, const cv::Mat &patch2)
{
    if (!initialized)
    {
        std::cerr << "MATLAB engine not initialized!" << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    try
    {
        // Convert cv::Mat to MATLAB arrays
        auto matlab_patch1 = cvMatToMatlabArray(patch1);
        auto matlab_patch2 = cvMatToMatlabArray(patch2);

        // Set variables in MATLAB workspace
        matlabPtr->setVariable(u"patch1", matlab_patch1);
        matlabPtr->setVariable(u"patch2", matlab_patch2);

        // Call MATLAB NCC function
        matlabPtr->eval(u"ncc_result = compute_ncc(patch1, patch2);");

        // Get result back
        auto result = matlabPtr->getVariable(u"ncc_result");
        auto typed_result = static_cast<matlab::data::TypedArray<double>>(result);

        return typed_result[0];
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in MATLAB NCC computation: " << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
}

matlab::data::TypedArray<double> MatlabNCCComputer::cvMatToMatlabArray(const cv::Mat &mat)
{
    cv::Mat mat_double;
    mat.convertTo(mat_double, CV_64F);

    // Create MATLAB array dimensions
    std::vector<size_t> dims = {static_cast<size_t>(mat.rows),
                                static_cast<size_t>(mat.cols)};

    // Create buffer for data
    matlab::data::buffer_ptr_t<double> dataPtr = factory->createBuffer<double>(mat.rows * mat.cols);

    // Copy data (OpenCV is row-major, MATLAB is column-major)
    double *buffer = dataPtr.get();
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            buffer[j * mat.rows + i] = mat_double.at<double>(i, j);
        }
    }

    return factory->createArrayFromBuffer<double>(dims, std::move(dataPtr));
}

// Global instance getter
MatlabNCCComputer &getMatlabNCCComputer()
{
    static MatlabNCCComputer instance;
    if (!instance.isInitialized())
    {
        instance.initialize();
    }
    return instance;
}

#endif // USE_MATLAB_NCC