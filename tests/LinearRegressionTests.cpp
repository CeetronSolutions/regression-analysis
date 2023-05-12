/////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2023-     Equinor ASA
//
//  roffcpp is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  roffcpp is distributed in the hope that it will be useful, but WITHOUT ANY
//  WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.
//
//  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
//  for more details.
//
/////////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "LinearRegression.hpp"

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( LinearRegressionTests, SimpleTest )
{
    // Test data generated with scikit-learn
    std::vector<double> x = { 1, 2, 3, 4, 5 };
    std::vector<double> y = { 10, 18, 33, 40, 51 };

    double expectedSlope     = 10.4;
    double expectedIntercept = -0.8;

    LinearRegression regression;
    regression.fit( x, y );
    ASSERT_NEAR( expectedSlope, regression.slope(), 0.00001 );
    ASSERT_NEAR( expectedIntercept, regression.intercept(), 0.00001 );

    std::vector<double> input = { 1.0, 1.2, 4.0, 4.3, 44.4 };

    std::vector<double> expectedValues;
    for ( auto v : input )
    {
        expectedValues.push_back( expectedSlope * v + expectedIntercept );
    }

    std::vector<double> predictedValues = regression.predict( input );
    for ( size_t i = 0; i < input.size(); i++ )
    {
        ASSERT_NEAR( expectedValues[i], predictedValues[i], 0.00001 );
    }
}
