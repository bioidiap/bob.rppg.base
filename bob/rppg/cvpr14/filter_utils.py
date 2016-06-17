#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Wed  4 Nov 09:58:53 CET 2015

import numpy

def detrend(signal, Lambda):
  """detrend(signal, Lambda) -> filtered_signal
  
  This function applies a detrending filter.
   
  This code is based on the following article "An advanced detrending method with application
  to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
  **Parameters**

    ``signal`` (1d numpy array):
      The signal where you want to remove the trend.

    ``Lambda`` (int):
      The smoothing parameter.

  **Returns**
  
    ``filtered_signal`` (1d numpy array):
      The detrended signal.
  """
  signal_length = signal.shape[0]

  # observation matrix
  H = numpy.identity(signal_length) 

  # second-order difference matrix
  from scipy.sparse import spdiags
  ones = numpy.ones(signal_length)
  minus_twos = -2*numpy.ones(signal_length)
  diags_data = numpy.array([ones, minus_twos, ones])
  diags_index = numpy.array([0, 1, 2])
  D = spdiags(diags_data, diags_index, (signal_length-2), signal_length).toarray()
  filtered_signal = numpy.dot((H - numpy.linalg.inv(H + (Lambda**2) * numpy.dot(D.T, D))), signal)
  return filtered_signal

def average(signal, window_size):
  """average(signal, window_size) -> filtered_signal
  
  Moving average filter.

  **Parameters**

    ``signal`` (1d numpy array):
      The signal to filter.

    ``window_size`` (int):
      The size of the window to compute the average.

  **Returns**
  
    ``filtered_signal`` (1d numpy array):
      The averaged signal.
  """
  from scipy.signal import lfilter
  a = 1.0 
  b = numpy.zeros(window_size)
  b += (1.0 / float(window_size))
  filtered_signal = lfilter(b, a, signal)
  return filtered_signal
