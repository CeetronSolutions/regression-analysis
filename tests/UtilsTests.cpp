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

#include "Utils.hpp"

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( UtilsTests, CoefficientOfDeterminationTest )
{
    // Test data generated with scikit-learn
    std::vector<double> actual     = { 3, -0.5, 2, 7 };
    std::vector<double> predicted  = { 2.5, 0.0, 2, 8 };
    double              expectedR2 = 0.9486081370449679;

    double r2 = Utils::computeR2( actual, predicted );
    ASSERT_NEAR( expectedR2, r2, 0.00001 );
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
TEST( UtilsTests, CoefficientOfDeterminationPerfectScoreTest )
{
    // Test data generated with scikit-learn
    std::vector<double> actual     = { 3, -0.5, 2, 7 };
    std::vector<double> predicted  = { 3, -0.5, 2, 7 };
    double              expectedR2 = 1.0;

    double r2 = Utils::computeR2( actual, predicted );
    ASSERT_NEAR( expectedR2, r2, 0.00001 );
}
