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

#include "PolynominalRegression.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
PolynominalRegression::PolynominalRegression()
{
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
void PolynominalRegression::fit( const std::vector<double>& x, const std::vector<double>& y, int degree )
{
    std::size_t n = x.size();

    // Create a matrix to hold the input data and a vector to hold the output data
    Eigen::MatrixXd X( n, degree + 1 );
    Eigen::VectorXd Y( n );

    // Fill the matrix and vector with data
    for ( std::size_t i = 0; i < n; i++ )
    {
        for ( std::size_t j = 0; j <= static_cast<size_t>( degree ); j++ )
        {
            X( i, j ) = std::pow( x[i], j );
        }

        Y( i ) = y[i];
    }

    // Solve for coeffisients
    Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve( Y );

    m_coeffisients.clear();
    for ( int i = 0; i < coeffs.size(); i++ )
    {
        m_coeffisients.push_back( coeffs( i ) );
    }
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
std::vector<double> PolynominalRegression::coeffisients() const
{
    return m_coeffisients;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
std::vector<double> PolynominalRegression::predict( const std::vector<double>& values ) const
{
    std::vector<double> predictedValues;

    for ( auto v : values )
        predictedValues.push_back( computePolynominal( v, m_coeffisients ) );

    return predictedValues;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double PolynominalRegression::computePolynominal( double input, const std::vector<double>& coeffisients )
{
    double result = 0.0;
    for ( size_t j = 0; j < coeffisients.size(); j++ )
    {
        result += coeffisients[j] * std::pow( input, j );
    }

    return result;
}
