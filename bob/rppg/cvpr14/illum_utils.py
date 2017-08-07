#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
# 
# This file is part of bob.rpgg.base.
# 
# bob.rppg.base is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# bob.rppg.base is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bob.rppg.base. If not, see <http://www.gnu.org/licenses/>.

import numpy

def rectify_illumination(face_color, bg_color, step, length):
  """rectify_illumination(face_color, bg_color, step, length) -> rectified color
  
  This function performs the illumination rectification.
  
  The correction is made on the face green values using the background green values, 
  so as to remove global illumination variations in the face green color signal.

  **Parameters**

    ``face_color`` (1d numpy array):
      The mean green value of the face across the video sequence. 

    ``bg_color`` (1d numpy array):
      The mean green value of the background across the video sequence. 

    ``step`` (float):
      Step size in the filter's weight adaptation.

    ``length`` (int):
      Length of the filter.

  **Returns**
    
    ``rectified color`` (1d numpy array):
      The mean green values of the face, corrected for illumination variations.
  """
  # first pass to find the filter coefficients
  # - y: filtered signal
  # - e: error (aka difference between face and background)
  # - w: filter coefficient(s)
  yg, eg, wg = nlms(bg_color, face_color, length, step)

  # second pass to actually filter the signal, using previous weights as initial conditions
  # the second pass just filters the signal and does NOT update the weights !
  yg2, eg2, wg2 = nlms(bg_color, face_color, length, step, initCoeffs=wg, adapt=False)
  return eg2

def nlms(signal, desired_signal, n_filter_taps, step, initCoeffs=None, adapt=True):
  """nlms(signal, desired_signal, n_filter_taps, step[, initCoeffs][, adapt]) -> y, e, w
  
  Normalized least mean square filter.
  
  Based on adaptfilt 0.2:  https://pypi.python.org/pypi/adaptfilt/0.2
  
  **Parameters**
  
    ``signal`` (1d numpy array):
      The signal to be filtered.
  
    ``desired_signal`` (1d numpy array):
      The target signal.

    ``n_filter_taps`` (int):
      The number of filter taps (related to the filter order).
  
    ``step`` (float):
      Adaptation step for the filter weights.
  

    ``initCoeffs`` ([Optional] numpy array (1, n_filter_taps)):
      Initial values for the weights. Defaults to zero.

    ``adapt`` ([Optional] boolean):
      If True, adapt the filter weights. If False, only filters.
      Defaults to True.

  **Returns**

    ``y`` (1d numpy array):
      The filtered signal.
    
    ``e`` (1d numpy array):
      The error signal (difference between filtered and desired)

    ``w`` (numpy array (1, n_filter_taps)):
      The found weights of the filter.
      
  """
  eps = 0.001
  number_of_iterations = len(signal) - n_filter_taps + 1
  if initCoeffs is None:
    initCoeffs = numpy.zeros(n_filter_taps)

  # Initialization
  y = numpy.zeros(number_of_iterations)    # Filter output
  e = numpy.zeros(number_of_iterations)    # Error signal
  w = initCoeffs                           # Initial filter coeffs

  # Perform filtering
  errors = []
  for n in range(number_of_iterations):
      x = numpy.flipud(signal[n:(n + n_filter_taps)])  # Slice to get view of M latest datapoints
      y[n] = numpy.dot(x, w)
      e[n] = desired_signal[n + n_filter_taps - 1] - y[n]
      errors.append(e[n])

      if adapt:
        normFactor = 1./(numpy.dot(x, x) + eps)
        w =  w + step * normFactor * x * e[n]
        y[n] = numpy.dot(x, w)

  return y, e, w
