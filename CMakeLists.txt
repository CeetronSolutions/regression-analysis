cmake_minimum_required(VERSION 3.15)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

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
  include(${CMAKE_TOOLCHAIN_FILE})
endif()

find_package(Eigen3 REQUIRED)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
