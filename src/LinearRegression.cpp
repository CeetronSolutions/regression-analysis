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

#include "LinearRegression.hpp"

#include <Eigen/Core>

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
    double learningRate  = 0.01;
    int    numIterations = 10000;

    auto copyData = []( const std::vector<double>& from, Eigen::VectorXd& to )
    {
        size_t numElements = from.size();
        for ( size_t i = 0; i < numElements; i++ )
            to[i] = from[i];
    };

    Eigen::VectorXd X( x.size() );
    copyData( x, X );

    Eigen::VectorXd Y( y.size() );
    copyData( y, Y );

    size_t n = x.size();

    m_slope     = 0.0;
    m_intercept = 0.0;

    // Find Gradient Descent
    for ( int i = 0; i < numIterations; i++ )
    {
        // Compute current prediction
        Eigen::VectorXd Y_pred = ( m_slope * X ).array() + m_intercept;
        // Compute derivative of slope
        double slopeDerivative = ( -2.0 / n ) * ( X.cwiseProduct( Y - Y_pred ) ).sum();
        // Compute derivative of the intercept
        double interceptDerivative = ( -2.0 / n ) * ( Y - Y_pred ).sum();

        m_slope     = m_slope - learningRate * slopeDerivative;
        m_intercept = m_intercept - learningRate * interceptDerivative;
    }
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
std::vector<double> LinearRegression::predict( const std::vector<double>& values ) const
{
    std::vector<double> predictedValues;
    for ( auto v : values )
        predictedValues.push_back( v * m_slope + m_intercept );

    return predictedValues;
}
