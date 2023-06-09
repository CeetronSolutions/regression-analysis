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
class LogisticRegression
{
public:
    LogisticRegression();

    void fit( const std::vector<double>& x, const std::vector<double>& y );

    std::vector<double> predict( const std::vector<double>& values ) const;

private:
    std::vector<double> m_theta;
};
} // namespace regression
