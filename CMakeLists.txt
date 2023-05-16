cmake_minimum_required(VERSION 3.15)

project(regression-analysis
  VERSION 0.1
  DESCRIPTION "Regression library for C++."
  LANGUAGES CXX)

option(REGRESSION_FETCH_EIGEN "Fetch Eigen" ON)
if(REGRESSION_FETCH_EIGEN)
  add_subdirectory(thirdparty)
endif()

find_package(Eigen3 REQUIRED)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
