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

    ``signal`` : (1d numpy array)
      The signal where you want to remove the trend.

    ``Lambda`` : (int)
      The smoothing parameter.

  **Returns**
  
    ``filtered_signal`` : (1d numpy array)
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

    ``signal`` : (1d numpy array)
      The signal to filter.

    ``window_size``: (int)
      The size of the window to compute the average.

  **Returns**
  
    ``filtered_signal`` : (1d numpy array)
      The averaged signal.
  """
  from scipy.signal import lfilter
  a = 1.0 
  b = numpy.zeros(window_size)
  b += (1.0 / float(window_size))
  filtered_signal = lfilter(b, a, signal)
  return filtered_signal

def build_bandpass_filter(fs, order, plot=False):
  """build_bandpass_filter(fs, order[, plot]) -> b
  
  Builds a butterworth bandpass filter.
  
  **Parameters**

    ``fs`` : (float)
      sampling frequency of the signal (i.e. framerate).
    
    ``order`` : (int)
      The order of the filter (the higher, the sharper).
  
    ``plot`` : ([Optional] boolean)
      Plots the frequency response of the filter.
      Defaults to False.
  
  **Returns**
    
    ``b`` : (numpy array)
      The coefficients of the FIR filter.
  """
  # frequency range in Hertz, corresponds to plausible h
  #heart-rate values, i.e. [42-240] beats per minute
  min_freq = 0.7 
  max_freq = 4.0 

  from scipy.signal import firwin 
  nyq = fs / 2.0
  numtaps = order + 1
  b = firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)

  # show the frequency response of the filter
  if plot:
    from matplotlib import pyplot
    from scipy.signal import freqz
    w, h = freqz(b)
    fig = pyplot.figure()
    pyplot.title('Bandpass filter frequency response')
    ax1 = fig.add_subplot(111)
    pyplot.plot(w * fs / (2 * numpy.pi), 20 * numpy.log10(abs(h)), 'b')
    [ymin, ymax] = ax1.get_ylim()
    pyplot.vlines(min_freq, ymin, ymax, color='red', linewidths='2')
    pyplot.vlines(max_freq, ymin, ymax, color='red', linewidths='2')
    pyplot.ylabel('Amplitude [dB]', color='b')
    pyplot.xlabel('Frequency [Hz]')
    pyplot.show()

  return b
