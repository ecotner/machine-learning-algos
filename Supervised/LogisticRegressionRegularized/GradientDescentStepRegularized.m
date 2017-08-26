## Copyright (C) 2017 27182_000
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} GradientDescent (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 27182_000 <27182_000@ALPHAMACHINE>
## Created: 2017-08-02

function theta = GradientDescentStepRegularized (X, y, theta, alpha, lambda)
     m = length(y);
     n = length(theta) - 1;
     theta = theta*(1 - (alpha*lambda/m)) - (alpha/m)*(X'*(1./(1 + e.^(-X*theta)) - y));
     % update rules if theta(1) is excluded from the cost function:
%     theta(1) = theta(1) - (alpha/m)*(X'*(1./(1 + e.^(-X*theta)) - y))(1);
%     theta(2:n+1) = theta(2:n+1)*(1 - (alpha*lambda/m)) - (alpha/m)*(X'*(1./(1 + e.^(-X*theta)) - y))(2:n+1);

endfunction
