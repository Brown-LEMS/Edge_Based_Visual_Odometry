set (control_sources
        Frame.cpp
        Dataset.cpp
        Pipeline.cpp
        utility.cpp
        MotionTracker.cpp
        toed/cpu_toed.cpp
        Stereo_Matches.cpp
        Temporal_Matches.cpp
        Stereo_Iterator.cpp
        EdgeClusterer.cpp
)

add_library( lems_vo_slam ${control_sources} )
include_directories(/gpfs/data/bkimia/opencv-4.x/build_install/include/opencv4)

#> Linking third-party libraries
target_link_libraries(lems_vo_slam
    PUBLIC OpenMP::OpenMP_CXX
    -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
    ${THIRD_PARTY_LIBS}
)

# Add optional glog and gflags linking if found and enabled
if(USE_GLOGS AND GLOG_FOUND AND GLOG_LIBRARIES)
    target_link_libraries(lems_vo_slam PUBLIC ${GLOG_LIBRARIES})
    if(GLOG_INCLUDE_DIRS)
        target_include_directories(lems_vo_slam PUBLIC ${GLOG_INCLUDE_DIRS})
    endif()
endif()

if(USE_GLOGS AND GFLAGS_FOUND AND GFLAGS_LIBRARIES)
    target_link_libraries(lems_vo_slam PUBLIC ${GFLAGS_LIBRARIES})
    if(GFLAGS_INCLUDE_DIRS)
        target_include_directories(lems_vo_slam PUBLIC ${GFLAGS_INCLUDE_DIRS})
    endif()
endif()