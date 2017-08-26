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
## @deftypefn {} {@var{retval} =} costFunctionJ (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 27182_000 <27182_000@ALPHAMACHINE>
## Created: 2017-07-31

function J = costFunctionLogisticRegularized (X, y, theta, lambda)
     m = length(y);
     z = X*theta;
     J = (1/m) * (y'*log(1 + e.^(-z)) + (1-y)'*log(1 + e.^z)) + (lambda/(2*m))*norm(theta)^2;
     % cost function if theta(1) is excluded from the regularization:    
%     J = (1/m) * (y'*log(1 + e.^(-z)) + (1-y)'*log(1 + e.^z)) + (lambda/(2*m))*norm(theta(2:length(theta)))^2;

endfunction
