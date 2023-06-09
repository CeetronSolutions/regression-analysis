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

#include "LinearRegression.hpp"

#include "Utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
LinearRegression::LinearRegression()
    : m_slope( 0.0 )
    , m_intercept( 0.0 )
{
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
void LinearRegression::fit( const std::vector<double>& x, const std::vector<double>& y )
{
    std::size_t n = x.size();

    // Create a matrix to hold the input data and a vector to hold the output data
    Eigen::MatrixXd X( n, 2 );
    Eigen::VectorXd Y( n );

    // Fill the matrix and vector with data
    for ( std::size_t i = 0; i < n; i++ )
    {
        X( i, 0 ) = x[i];
        X( i, 1 ) = 1.0;
        Y( i )    = y[i];
    }

    // Use Eigen's QR decomposition to calculate the regression coefficients
    Eigen::VectorXd beta = X.colPivHouseholderQr().solve( Y );

    m_slope     = beta( 0 );
    m_intercept = beta( 1 );

    m_r2 = Utils::computeR2( y, predict( x ) );
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double LinearRegression::slope() const
{
    return m_slope;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double LinearRegression::intercept() const
{
    return m_intercept;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double LinearRegression::r2() const
{
    return m_r2;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
std::vector<double> LinearRegression::predict( const std::vector<double>& values ) const
{
    std::vector<double> predictedValues;
    for ( auto v : values )
        predictedValues.push_back( v * m_slope + m_intercept );

    return predictedValues;
}
