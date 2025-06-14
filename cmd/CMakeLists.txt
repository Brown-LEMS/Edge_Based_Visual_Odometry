add_executable(main_VO main_VO.cpp)

if(DEFINED SAUL_BUILD)
  target_link_libraries(main_VO
    lems_vo_slam
    ${THIRD_PARTY_LIBS}
  )
else()
  target_link_libraries(main_VO
    lems_vo_slam
    yaml-cpp
    ${THIRD_PARTY_LIBS}
  )
  link_directories(/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64)
endif()
