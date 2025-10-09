#ifndef MATLAB_NCC_COMPUTER_H
#define MATLAB_NCC_COMPUTER_H

#ifdef USE_MATLAB_NCC

#include <opencv2/opencv.hpp>
#include <memory>

// Forward declarations to avoid including MATLAB headers in header file
namespace matlab
{
    namespace engine
    {
        class MATLABEngine;
    }
    namespace data
    {
        class ArrayFactory;
        template <typename T>
        class TypedArray;
    }
}

class MatlabNCCComputer
{
private:
    std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
    std::unique_ptr<matlab::data::ArrayFactory> factory;
    bool initialized;

public:
    MatlabNCCComputer();
    ~MatlabNCCComputer();

    // Disable copy constructor and assignment operator
    MatlabNCCComputer(const MatlabNCCComputer &) = delete;
    MatlabNCCComputer &operator=(const MatlabNCCComputer &) = delete;

    bool initialize();
    double computeNCC(const cv::Mat &patch1, const cv::Mat &patch2);
    bool isInitialized() const { return initialized; }

private:
    matlab::data::TypedArray<double> cvMatToMatlabArray(const cv::Mat &mat);
};

// Global instance getter
MatlabNCCComputer &getMatlabNCCComputer();

#endif // USE_MATLAB_NCC

#endif // MATLAB_NCC_COMPUTER_H