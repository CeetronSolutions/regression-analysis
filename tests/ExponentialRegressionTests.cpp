/////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2023-     Equinor ASA
//
//  regression-analysis is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  regression-analysis is distributed in the hope that it will be useful, but WITHOUT ANY
//  WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.
//
//  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
//  for more details.
//
/////////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include <cmath>

#include "ExponentialRegression.hpp"

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( ExponentialRegressionTests, SimpleTest )
{
    // Test data generated python/numpy: np.polyfit(x, np.log(y), 1)
    std::vector<double> x = { 1, 2, 3, 4, 5 };
    std::vector<double> y = { 0.5, 1.2, 1.8, 2.3, 2.8 };

    double                expectedA = std::exp( -0.84093831 );
    double                expectedB = 0.40961208;
    ExponentialRegression regression;
    regression.fit( x, y );
    ASSERT_NEAR( expectedB, regression.b(), 0.00001 );
    ASSERT_NEAR( expectedA, regression.a(), 0.00001 );

    std::vector<double> input = { 1.0, 1.2, 4.0, 4.3, 44.4 };

    std::vector<double> expectedValues;
    for ( auto v : input )
    {
        expectedValues.push_back( std::exp( v * expectedB ) * expectedA );
    }

    std::vector<double> predictedValues = regression.predict( input );
    for ( size_t i = 0; i < input.size(); i++ )
    {
        ASSERT_NEAR( expectedValues[i], predictedValues[i], expectedValues[i] * 0.0000002 );
    }
}
