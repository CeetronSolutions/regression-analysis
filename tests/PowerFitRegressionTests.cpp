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

#include "PowerFitRegression.hpp"

#include <cmath>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( PowerFitRegressionTests, SimpleTest )
{
    std::vector<double> x = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> y = { 1.0, 4.0, 9.0, 16.0, 25.0 };

    double expectedExponent = 2.0;
    double expectedScale    = 1.0;

    PowerFitRegression regression;
    regression.fit( x, y );
    ASSERT_NEAR( expectedExponent, regression.exponent(), 0.00001 );
    ASSERT_NEAR( expectedScale, regression.scale(), 0.00001 );

    std::vector<double> input = { 1.0, 1.2, 4.0, 4.3, 10.4 };

    std::vector<double> expectedValues;
    for ( auto v : input )
    {
        expectedValues.push_back( regression.scale() * std::pow( v, regression.exponent() ) );
    }

    std::vector<double> predictedValues = regression.predict( input );
    for ( size_t i = 0; i < input.size(); i++ )
    {
        ASSERT_NEAR( expectedValues[i], predictedValues[i], 0.00001 );
    }
}
