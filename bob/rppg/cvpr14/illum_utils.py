#!/usr/bin/env python
# encoding: utf-8

import numpy

def rectify_illumination(face_color, bg_color, step, length):
  """performs illumination rectification.
  
  The correction is made on the face green values using the background green values, 
  so as to remove global illumination variations in the face green color signal.

  Parameters
  ----------
  face_color: numpy.ndarray
    The mean green value of the face across the video sequence. 
  bg_color: numpy.ndarray
    The mean green value of the background across the video sequence. 
  step: float
    Step size in the filter's weight adaptation.
  length: int
    Length of the filter.

  Returns
  -------
  rectified color: numpy.ndarray
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
  """Normalized least mean square filter.
  
  Based on adaptfilt 0.2:  https://pypi.python.org/pypi/adaptfilt/0.2
  
  Parameters
  ----------
  signal: numpy.ndarray
    The signal to be filtered.
  desired_signal: numpy.ndarray
    The target signal.
  n_filter_taps: int
    The number of filter taps (related to the filter order).
  step: float
    Adaptation step for the filter weights.
  initCoeffs: numpy.ndarray 
    Initial values for the weights. Defaults to zero.
  adapt: bool
    If True, adapt the filter weights. If False, only filters.

  Returns
  -------
  y: numpy.ndarray
    The filtered signal.
    
  e: numpy.ndarray
    The error signal (difference between filtered and desired)

  w: numpy.ndarray
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
