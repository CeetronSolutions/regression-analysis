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

#include "PolynomialRegression.hpp"

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( PolynomialRegressionTests, SimpleTest )
{
    // Test data generated with scikit-learn
    std::vector<double> x              = { 2, 3, 4, 5, 6, 7, 7, 8, 9, 11, 12 };
    std::vector<double> y              = { 18, 16, 15, 17, 20, 23, 25, 28, 31, 30, 29 };
    std::vector<double> expectedCoeffs = { 33.626400375322575, -11.83877127, 2.25592957, -0.10889554 };
    std::vector<double> input          = { 1.0, 1.2, 4.0, 4.3, 44.4 };
    std::vector<double> expectedValues = { 23.93466313, 22.48024193, 15.39687368, 15.77386375, -5576.21655925 };

    PolynomialRegression regression;
    regression.fit( x, y, 3 );

    std::vector<double> actualCoeffs = regression.coeffisients();
    ASSERT_EQ( expectedCoeffs.size(), actualCoeffs.size() );
    for ( size_t i = 0; i < expectedCoeffs.size(); i++ )
    {
        ASSERT_NEAR( expectedCoeffs[i], actualCoeffs[i], 0.0001 );
    }

    std::vector<double> predictedValues = regression.predict( input );
    for ( size_t i = 0; i < input.size(); i++ )
    {
        ASSERT_NEAR( expectedValues[i], predictedValues[i], 0.00001 );
    }
}
