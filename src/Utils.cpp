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

#include "Utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double Utils::computeR2( const std::vector<double>& actualY, const std::vector<double>& predictedY )
{
    assert( actualY.size() == predictedY.size() );

    int             n = static_cast<int>( actualY.size() );
    Eigen::VectorXd Y( n );
    Eigen::VectorXd Ypred( n );

    for ( size_t i = 0; i < actualY.size(); i++ )
    {
        Y( i )     = actualY[i];
        Ypred( i ) = predictedY[i];
    }

    double actualYMean = Y.mean();
    // Compute the total sum of squares (TSS) which represents the total variability in the target.
    double tss = ( Y.array() - actualYMean ).square().sum();
    // Compute the residual sum of squares (RSS), which represents the variability not explained by the regression.
    double rss = ( Y.array() - Ypred.array() ).square().sum();

    // The coefficient of determination (R2) represents the proportion of the total variability in the target values
    // that is explained by the regression. A value close to 1 indicates a good fit, while a value close to 0 suggests
    // that the regression does not explain much of the variability,
    return 1.0 - rss / tss;
}
