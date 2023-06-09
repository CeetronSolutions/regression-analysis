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

#include "LogisticRegression.hpp"

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( LogisticRegressionTests, SimpleTest )
{
    // Test data taken from: https://realpython.com/logistic-regression-python/
    std::vector<double> x = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<double> y = { 0, 1, 0, 0, 1, 1, 1, 1, 1, 1 };

    LogisticRegression regression;
    regression.fit( x, y );

    std::vector<double> predictedValues = regression.predict( x );
    std::vector<double> expectedValues =
        { 0.12208792, 0.24041529, 0.41872657, 0.62114189, 0.78864861, 0.89465521, 0.95080891, 0.97777369, 0.99011108, 0.99563083 };

    for ( size_t i = 0; i < x.size(); i++ )
    {
        ASSERT_NEAR( expectedValues[i], predictedValues[i], 0.00001 );
    }
}
