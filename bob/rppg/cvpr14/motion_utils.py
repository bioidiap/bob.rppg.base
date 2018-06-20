#!/usr/bin/env python
# encoding: utf-8

import numpy

def build_segments(signal, length):
  """builds an array containing segments of the signal.

  The signal is divided into segments of provided length
  (no overlap) and the different segments are stacked.

  Parameters
  ----------
  signal: numpy.ndarray
    The signal to be processed.
  length: int
    The length of the segments.

  Returns
  -------
  segments: numpy.ndarray 
    the segments composing the signal.
  end_index: int
    The length of the signal (there may be a trail smaller
    than a segment at the end of the signal, that will
    be discarded).
 
  """
  number_of_segments = int(numpy.floor(signal.shape[0] /  float(length)))
  end_index = number_of_segments * length 
  segments = numpy.reshape(signal[:end_index], (number_of_segments, length))
  return segments, end_index

def prune_segments(segments, threshold):
  """remove segments.

  Segments are removed if their standard deviation is higher than
  the provided threshold.

  Parameters
  ----------
  segments: numpy.ndarray
    The set of segments.
  threshold: float
    Threshold on the standard deviation.

  Returns
  -------
  pruned_segments: numpy.ndarray
    The set of "stable" segments.
  gaps: list of length (# of retained segments)
    Boolean list that tells if a gap should be accounted for
    when building the final signal.
  cut_index: list of tuples
    Contains the start and end index of each removed segment.
    Used for plotting purposes.
  
  """
  final_segments = []
  gaps = [] # the first kept segment could not have a gap
  cut_index = [] 
  for i in range(segments.shape[0]):
    # if this segment is below the threshold, keep it
    if numpy.std(segments[i], ddof=1) <= threshold:
      final_segments.append(segments[i])
      # if this is the first segment, no need to take care of the gap
      if len(final_segments) == 1:
        gaps.append(False)
      # if this is not the first segment, and that previous one was discarded, gap
      elif numpy.std(segments[i-1], ddof=1) > threshold :
        gaps.append(True)
      # this is not the first, but the previous one has not been discarded
      else:
        gaps.append(False)
    else:
      cut_index.append(((i*segments.shape[1]), ((i+1)*segments.shape[1])))
  return numpy.array(final_segments), gaps, cut_index

def build_final_signal(segments, gaps):
  """builds the final signal with remaining segments.

  Parameters
  ----------
  segments: numpy.ndarray
    The set of remaining segments.
  gaps: list
    Boolean list that tells if a gap should be accounted for
    when building the final signal.
  
  Returns
  -------
  final_signal: numpy.ndarray
    The final signal.
  
  """
  signal_length = segments.shape[0] * segments.shape[1]
  final_signal = numpy.zeros(signal_length)
  gap = 0
  for i in range(segments.shape[0]):
    final_signal[i*segments.shape[1]: (i+1)*segments.shape[1]] = segments[i]
    # fill the vertical gap
    if gaps[i]:
      # compute the gap between this segment and the previous one
      gap = segments[i, 0] - segments[i-1, -1]
      # correct all of the following segments (they will be shifted)
      segments[i:, :] -= gap
      # build the final signal using the shifted segment
      final_signal[i*segments.shape[1]:(i+1)*segments.shape[1]] = segments[i]
  return final_signal

def build_final_signal_cvpr14(segments, gaps):
  """builds the final signal with remaining segments.
 
  .. WARNING::
     This contains a bug !
  
  Builds the final signal, but reproducing the bug found
  in the code provided by the authors of [li-cvpr-2014]_.
  The bug is in the 'collage' of remaining segments. The
  gap is not always properly accounted for... 

  Parameters
  ----------
  segments: numpy.ndarray
    The set of remaining segments.
  gaps: list
    Boolean list that tells if a gap should be accounted for
    when building the final signal.
  
  Returns
  -------
  final_signal: numpy.ndarray
    The final signal.
  
  """
  signal_length = segments.shape[0] * segments.shape[1]
  final_signal = numpy.zeros(signal_length)
  gap = 0
  original_segments = numpy.copy(segments)
  for i in range(segments.shape[0]):
    final_signal[i*segments.shape[1]: (i+1)*segments.shape[1]] = segments[i]
    # fill the vertical gap
    if gaps[i]:
      # compute the gap between this segment and the previous one
      # XXX the bug is here: gap is computed using the original signal
      # XXX instead of the corrected one (if there was one or more previous gaps)
      gap = segments[i, 0] - original_segments[i-1, -1]
      # correct all of the following segments (they will be shifted)
      segments[i:, :] -= gap
      # build the final signal using the shifted segment
      final_signal[i*segments.shape[1]:(i+1)*segments.shape[1]] = segments[i]
  return final_signal

