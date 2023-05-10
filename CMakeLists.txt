cmake_minimum_required(VERSION 3.15)


project(regression-analysis
  VERSION 0.1
  DESCRIPTION "Regression library for C++."
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(REGRESSION_ANALYSIS_VCPKG_AUTO_INSTALL "Automatically download build pre-requisites with VCPKG" OFF)

if(REGRESSION_ANALYSIS_VCPKG_AUTO_INSTALL)
  include(AutoVcpkg)
  set(AUTO_VCPKG_ROOT "${CMAKE_SOURCE_DIR}/vcpkg")
  vcpkg_install(eigen3)
endif()

find_package(Eigen3 REQUIRED)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
