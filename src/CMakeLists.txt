set (control_sources
        Frame.cpp
        Dataset.cpp
        Pipeline.cpp
        utility.cpp
        MotionTracker.cpp
        toed/cpu_toed.cpp
)

add_library( lems_vo_slam ${control_sources} )
include_directories(/gpfs/data/bkimia/cchien3/opencv-4.7.0/opencv_install/include/opencv4)

#> Linking third-party libraries
target_link_libraries(lems_vo_slam
        PUBLIC OpenMP::OpenMP_CXX
        -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
        -L/gpfs/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/boost-1.80.0-harukoycghoq6exyzhd3jonvup3gpqeo/lib boost_filesystem
        ${THIRD_PARTY_LIBS}
)