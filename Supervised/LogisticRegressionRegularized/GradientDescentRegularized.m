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

function [J_hist, theta] = GradientDescentRegularized (X, y, theta, learning_rate, lambda, num_steps)
     J_hist = [];
     for i = 1:num_steps;
          theta = GradientDescentStepRegularized(X, y, theta, learning_rate, lambda);
          J = costFunctionLogisticRegularized(X, y, theta, lambda);
          J_hist = [J_hist, J];
     endfor;     

endfunction
