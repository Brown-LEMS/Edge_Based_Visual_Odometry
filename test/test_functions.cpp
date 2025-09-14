
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "../include/definitions.h"

#include "test_include/test_third_order_edges.hpp"
#include "test_include/test_NCC.hpp"

//> Activate the tests here
#define TEST_TOED_EDGES             (false)
#define TEST_NCC                    (true)
#define TEST_GRADIENT_DEPTHS        (false)
#define TEST_RELATIVE_POSE          (false)

int main(int argc, char **argv)
{
#if TEST_TOED_EDGES
    f_TEST_TOED();
#endif

#if TEST_NCC
    f_TEST_NCC();
#endif

#if TEST_RELATIVE_POSE
    f_TEST_RELATIVE_POSE();
#endif

    return 0;
}