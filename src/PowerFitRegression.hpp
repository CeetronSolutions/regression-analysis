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

#pragma once

#include <vector>

namespace regression
{
class PowerFitRegression
{
public:
    PowerFitRegression();

    void   fit( const std::vector<double>& x, const std::vector<double>& y );
    double scale() const;
    double exponent() const;

    std::vector<double> predict( const std::vector<double>& values ) const;

    double r2() const;

private:
    double m_scale;
    double m_exponent;
    double m_r2;
};
} // namespace regression
