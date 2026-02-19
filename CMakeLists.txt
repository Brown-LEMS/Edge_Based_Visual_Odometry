CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD "17")
# set(CMAKE_CXX_STANDARD "11")
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

PROJECT(EDGE_BASED_VO)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(SAUL_BUILD 1)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "" FORCE)

#> Crucial flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" CACHE STRING "" FORCE)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#> Dependencies
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
# FIND_PACKAGE(yaml-cpp REQUIRED)

#> Optional dependencies - glog and gflags
FIND_PACKAGE(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GLOG QUIET libglog)
    pkg_check_modules(GFLAGS QUIET libgflags)
endif()

# Try to find glog using find_package if pkg-config didn't find it
if(NOT GLOG_FOUND)
    find_package(glog QUIET)
    if(glog_FOUND)
        set(GLOG_FOUND TRUE)
        set(GLOG_LIBRARIES glog::glog)
        set(GLOG_INCLUDE_DIRS ${glog_INCLUDE_DIRS})
    endif()
endif()

# Try to find gflags using find_package if pkg-config didn't find it
if(NOT GFLAGS_FOUND)
    find_package(gflags QUIET)
    if(gflags_FOUND)
        set(GFLAGS_FOUND TRUE)
        set(GFLAGS_LIBRARIES gflags::gflags)
        set(GFLAGS_INCLUDE_DIRS ${gflags_INCLUDE_DIRS})
    endif()
endif()

# Check for local installations in the project directory
# Note: The local glog installation is incomplete, so we skip it for now
# Users can install glog/gflags system-wide or use -DUSE_GLOGS=0 to disable
if(NOT GLOG_FOUND)
    if(EXISTS "/gpfs/data/bkimia/cchien3/glog")
        message(STATUS "Found local glog installation but it's incomplete - skipping")
        message(STATUS "To use glog/gflags, install them system-wide or use -DUSE_GLOGS=0 to disable")
    endif()
endif()

if(NOT GFLAGS_FOUND)
    if(EXISTS "/gpfs/data/bkimia/cchien3/gflags_install")
        set(GFLAGS_FOUND TRUE)
        set(GFLAGS_LIBRARIES gflags::gflags)
        set(GFLAGS_INCLUDE_DIRS "/gpfs/data/bkimia/cchien3/gflags_install/include")
        message(STATUS "Found local gflags installation")
    endif()
endif()

# Initialize empty variables if not found
if(NOT GLOG_FOUND)
    set(GLOG_LIBRARIES "")
    set(GLOG_INCLUDE_DIRS "")
endif()

if(NOT GFLAGS_FOUND)
    set(GFLAGS_LIBRARIES "")
    set(GFLAGS_INCLUDE_DIRS "")
endif()

# Set USE_GLOGS based on availability and user preference
if(DEFINED USE_GLOGS)
    # User explicitly set USE_GLOGS, respect their choice
    if(USE_GLOGS AND GLOG_FOUND AND GFLAGS_FOUND)
        message(STATUS "User requested glog/gflags and libraries found - enabling optional logging support")
        add_definitions(-DUSE_GLOGS=1)
    else()
        message(STATUS "User requested glog/gflags disabled or libraries not found - disabling optional logging support")
        set(USE_GLOGS FALSE)
        add_definitions(-DUSE_GLOGS=0)
    endif()
else()
    # Auto-detect based on availability
    if(GLOG_FOUND AND GFLAGS_FOUND)
        set(USE_GLOGS TRUE)
        message(STATUS "Found glog and gflags - enabling optional logging support")
        add_definitions(-DUSE_GLOGS=1)
    else()
        set(USE_GLOGS FALSE)
        message(STATUS "glog and/or gflags not found - disabling optional logging support")
        add_definitions(-DUSE_GLOGS=0)
    endif()
endif()

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
      pthread
      ${CSparse_LIBRARY}
      /usr/local/lib/libyaml-cpp.so
    )
  # Add glog and gflags only if found and enabled
  if(USE_GLOGS AND GLOG_FOUND AND GLOG_LIBRARIES)
    list(APPEND THIRD_PARTY_LIBS ${GLOG_LIBRARIES})
  endif()
  if(USE_GLOGS AND GFLAGS_FOUND AND GFLAGS_LIBRARIES)
    list(APPEND THIRD_PARTY_LIBS ${GFLAGS_LIBRARIES})
  endif()
  
  include_directories(/usr/local/include/yaml-cpp)
  include_directories(${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
  if(USE_GLOGS AND GLOG_FOUND)
    include_directories(${GLOG_INCLUDE_DIRS})
  endif()
  if(USE_GLOGS AND GFLAGS_FOUND)
    include_directories(${GFLAGS_INCLUDE_DIRS})
  endif()
else()
  #> for Brown CCV Oscars
  set(THIRD_PARTY_LIBS
      ${OpenCV_LIBS}
      ${Eigen_LIBS}
      ${Boost_LIBS}
      /gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64/
    )
  # Add glog and gflags only if found and enabled
  if(USE_GLOGS AND GLOG_FOUND AND GLOG_LIBRARIES)
    list(APPEND THIRD_PARTY_LIBS ${GLOG_LIBRARIES})
  endif()
  if(USE_GLOGS AND GFLAGS_FOUND AND GFLAGS_LIBRARIES)
    list(APPEND THIRD_PARTY_LIBS ${GFLAGS_LIBRARIES})
  endif()
  
  include_directories(/gpfs/data/bkimia/cchien3/yaml-cpp/bin/include)
  include_directories(${EIGEN3_INCLUDE_DIR})
  include_directories(/gpfs/data/bkimia/cchien3/opencv-4.7.0/opencv_install/include/opencv4)
  link_directories(/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64)
  if(USE_GLOGS AND GLOG_FOUND)
    include_directories(${GLOG_INCLUDE_DIRS})
  endif()
  if(USE_GLOGS AND GFLAGS_FOUND)
    include_directories(${GFLAGS_INCLUDE_DIRS})
  endif()
endif()

enable_testing()

#> All header files
include_directories(${PROJECT_SOURCE_DIR}/include)

#> Subdirectories
add_subdirectory(src)

add_subdirectory(cmd)
add_subdirectory(test)