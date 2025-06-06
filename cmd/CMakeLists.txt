add_executable(main_VO main_VO.cpp)
target_link_libraries(main_VO lems_vo_slam 
                      -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
                      ${THIRD_PARTY_LIBS}
)