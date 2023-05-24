

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)


set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set(THREADS_PREFER_PTHREAD_FLAG ON)


# Tests need to be added as executables first
add_executable(regression-analysis-tests LinearRegressionTests.cpp PolynominalRegressionTests.cpp PowerFitRegressionTests.cpp LogarithmicRegressionTests.cpp UtilsTests.cpp)


target_include_directories(regression-analysis-tests PUBLIC ../src/)

if(MSVC)
  target_compile_options(regression-analysis-tests PRIVATE /W4 /WX /wd4127)
else()
  target_compile_options(regression-analysis-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()


target_compile_features(regression-analysis-tests PRIVATE cxx_std_17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(regression-analysis-tests PRIVATE regression-analysis gtest gtest_main)
add_test(NAME regression-analysis-tests COMMAND regression-analysis-tests)
