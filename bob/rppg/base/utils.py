#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Tue 31 May 12:08:11 CEST 2016

import os, sys
import numpy
import collections

import bob.ip.base

Point = collections.namedtuple('Point', 'y,x')
BoundingBox = collections.namedtuple('BoundingBox', 'topleft,size,quality')

def load_bbox(fname):
  """load_bbox(fname) -> bounding_boxes
  Load bounding boxes from file.

  This function loads bounding boxes for each frame of a video sequence.

  **Parameters**

    ``fname`` : (string)
      Filename of the file containing the bounding boxes.

  **Returns**

    ``bounding_boxes``: (dict of BoundingBox)
      Dictionary of BoundingBox, the key is the frame number
  """
  retval = {}
  with open(fname, 'rt') as f:
    for row in f:
      if not row.strip(): continue #empty
      p = row.split()
      retval[int(p[0])] = BoundingBox(
          Point(int(p[2]), int(p[1])), #y, x
          Point(int(p[4]), int(p[3])), #height, width
          float(p[5]), #quality
          )
  return retval


def scale_image(image, width, height):
  """scale_image(image, width, height) -> scaled_image
  
  This function scales an image.

  **Parameters**
  
    ``image`` : (3d numpy array)
      The image to be scaled.
  
    ``width`` : (int)
      The new image width.
  
    ``height``: (int)
      The new image height

  **Returns**
  
    ``result`` : (3d numpy array)
      The scaled image
  """
  assert len(image.shape) == 3, "This is meant to work with color images (3 channels)"
  result = numpy.zeros((3, width, height))
  bob.ip.base.scale(image, result)
  return result


def crop_face(image, bbx, facewidth):
  """crop_face(image, bbx, facewidth) -> face
  
  This function crops a face from an image.
  
  **Parameters**
  
    ``image`` : (3d numpy array )
      The image containing the face.

    ``bbx`` : (BoundingBox)
      The bounding box of the face.

    ``facewidth``: (int)
      The width of the face after cropping.

  **Returns**
    
    ``face`` : (numpy array)
      The face image.
  """

  # TODO: should be changed to use regular 
  # BoundingBox class, and not the namedtuple
  # made by André ... 
  face = image[:, bbx.topleft.y:(bbx.topleft.y + bbx.size.y), bbx.topleft.x:(bbx.topleft.x + bbx.size.x)]
  aspect_ratio = bbx.size.y / bbx.size.x # height/width
  # TODO: bug with the aspect ratio, should be converted to float !! 
  faceheight = facewidth * aspect_ratio
  face = scale_image(face, faceheight, facewidth)
  face = face.astype('uint8')
  return face


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
