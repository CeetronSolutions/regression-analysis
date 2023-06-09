set(HEADER_LIST "LinearRegression.hpp" "PolynomialRegression.hpp" "PowerFitRegression.hpp" "LogarithmicRegression.hpp" "ExponentialRegression.hpp" "LogisticRegression.hpp" "Utils.hpp")
set(SOURCE_LIST "LinearRegression.cpp" "PolynomialRegression.cpp" "PowerFitRegression.cpp" "LogarithmicRegression.cpp" "ExponentialRegression.cpp" "LogisticRegression.cpp" "Utils.cpp")

add_library(regression-analysis ${SOURCE_LIST} ${HEADER_LIST})

if(MSVC)
  target_compile_options(regression-analysis PRIVATE /W4 /WX /wd4127 /wd5054)
else()
  target_compile_options(regression-analysis PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_include_directories(regression-analysis PUBLIC . )
target_link_libraries(regression-analysis Eigen3::Eigen)

target_compile_features(regression-analysis PUBLIC cxx_std_17)
