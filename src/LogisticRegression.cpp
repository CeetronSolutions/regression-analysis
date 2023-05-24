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

#include "LogisticRegression.hpp"

#include "Utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
LogisticRegression::LogisticRegression()
{
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
void LogisticRegression::fit( const std::vector<double>& x, const std::vector<double>& y )
{
    std::size_t n = x.size();

    // Create a matrix to hold the input data and a vector to hold the output data
    Eigen::MatrixXd X( n, 2 );
    Eigen::VectorXd Y( n );

    // Fill the matrix and vector with data
    for ( std::size_t i = 0; i < n; i++ )
    {
        X( i, 0 ) = 1.0;
        X( i, 1 ) = x[i];
        Y( i )    = y[i];
    }

    // Initialize theta vector with zeros
    Eigen::VectorXd theta = Eigen::VectorXd::Zero( 2 );

    // Compute the logistic regression parameters using gradient descent
    double learningRate  = 0.1;
    int    numIterations = 10000;
    for ( int iteration = 0; iteration < numIterations; ++iteration )
    {
        Eigen::VectorXd h        = 1.0 / ( 1.0 + ( -X * theta ).array().exp() );
        Eigen::VectorXd gradient = X.transpose() * ( h - Y ) / n;
        theta -= learningRate * gradient;
    }

    m_theta.clear();
    for ( int i = 0; i < theta.size(); i++ )
    {
        m_theta.push_back( theta( i ) );
    }
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
std::vector<double> LogisticRegression::predict( const std::vector<double>& values ) const
{
    std::size_t n = values.size();

    Eigen::MatrixXd X( n, 2 );

    // Fill the matrix and vector with data
    for ( std::size_t i = 0; i < n; i++ )
    {
        X( i, 0 ) = 1.0;
        X( i, 1 ) = values[i];
    }

    Eigen::VectorXd theta( m_theta.size() );
    for ( std::size_t i = 0; i < m_theta.size(); i++ )
        theta( i ) = m_theta[i];

    Eigen::VectorXd probabilities = 1.0 / ( 1.0 + ( -X * theta ).array().exp() );

    std::vector<double> probs( probabilities.size() );
    for ( size_t i = 0; i < n; i++ )
    {
        probs[i] = probabilities( i );
    }

    return probs;
}
