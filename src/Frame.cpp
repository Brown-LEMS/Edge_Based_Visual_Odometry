#ifndef FRAME_CPP
#define FRAME_CPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "Frame.h"

// =======================================================================================================
// class Frame: structurize data associate to a frame/camera/view
//
// ChangeLogs
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

Frame::Frame(long id, double time_stamp, Eigen::Matrix3d Rotation, Eigen::Vector3d Translation, cv::Mat Img)
    :ID(id), time_stamp(time_stamp), Abs_Rot(Rotation), Abs_Transl(Translation), Image(Img)
{

}

Frame::Ptr Frame::Create_Frame() {
    //static long ID_ = 0;
    Frame::Ptr new_frame(new Frame);
    //new_frame->ID = ID_++;
    return new_frame;
}





#endif