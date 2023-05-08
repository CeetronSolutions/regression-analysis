cmake_minimum_required(VERSION 3.15)

project(regression-analysis
  VERSION 0.1
  DESCRIPTION "Regression library for C++."
  LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
