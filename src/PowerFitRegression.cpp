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

#include "PowerFitRegression.hpp"
#include "LinearRegression.hpp"
#include "Utils.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace regression;

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
PowerFitRegression::PowerFitRegression()
    : m_scale( 0.0 )
    , m_exponent( 0.0 )
{
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
void PowerFitRegression::fit( const std::vector<double>& x, const std::vector<double>& y )
{
    auto computeLog = []( const std::vector<double>& x )
    {
        std::vector<double> xLog( x.size() );
        std::transform( x.begin(), x.end(), xLog.begin(), []( double val ) { return std::log( val ); } );
        return xLog;
    };

    std::vector<double> xLog = computeLog( x );
    std::vector<double> yLog = computeLog( y );

    LinearRegression reg;
    reg.fit( xLog, yLog );

    m_scale    = std::exp( reg.intercept() );
    m_exponent = reg.slope();

    m_r2 = Utils::computeR2( y, predict( x ) );
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double PowerFitRegression::scale() const
{
    return m_scale;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double PowerFitRegression::exponent() const
{
    return m_exponent;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
double PowerFitRegression::r2() const
{
    return m_r2;
}

//--------------------------------------------------------------------------------------------------
///
//--------------------------------------------------------------------------------------------------
std::vector<double> PowerFitRegression::predict( const std::vector<double>& values ) const
{
    std::vector<double> predictedValues;
    for ( auto v : values )
        predictedValues.push_back( scale() * std::pow( v, exponent() ) );

    return predictedValues;
}
