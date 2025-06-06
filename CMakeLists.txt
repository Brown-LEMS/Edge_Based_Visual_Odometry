CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD "17")
# set(CMAKE_CXX_STANDARD "11")
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

PROJECT(EDGE_BASED_VO)

# set(SAUL_BUILD 1)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "" FORCE)

#> Crucial flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" CACHE STRING "" FORCE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#> Dependencies
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
# FIND_PACKAGE(yaml-cpp REQUIRED)

message("\n=====OpenCV Library=====")
message(STATUS "[Directory]:    ${OpenCV_DIR}")
message(STATUS "[Version]:      ${OpenCV_VERSION}")
message(STATUS "[Libraries]:    ${OpenCV_LIBS}")
message(STATUS "[Include Path]: ${OpenCV_INCLUDE_DIRS}")
message( "========================\n")

if(DEFINED SAUL_BUILD)
  #> for MACOS
  set(THIRD_PARTY_LIBS
      ${OpenCV_LIBS}
      ${Eigen_LIBS}        
      ${GLOG_LIBRARIES}
      ${GFLAGS_LIBRARIES}
      pthread
      ${CSparse_LIBRARY}
      /usr/local/lib/libyaml-cpp.so
    )
  include_directories(/usr/local/include/yaml-cpp)
  include_directories(${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
else()
  #> for Brown CCV Oscars
  set(THIRD_PARTY_LIBS
      ${OpenCV_LIBS}
      ${Eigen_LIBS}
      ${Boost_LIBS}
      /gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64/
    )
  include_directories(/gpfs/data/bkimia/cchien3/yaml-cpp/bin/include)
  include_directories(${EIGEN3_INCLUDE_DIR})
  include_directories(/gpfs/data/bkimia/cchien3/opencv-4.7.0/opencv_install/include/opencv4)
  link_directories(/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64)
endif()

enable_testing()

#> All header files
include_directories(${PROJECT_SOURCE_DIR}/include)

#> Subdirectories
add_subdirectory(src)

add_subdirectory(cmd)
add_subdirectory(test)