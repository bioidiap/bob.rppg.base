#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Tue 31 May 12:08:11 CEST 2016

import os, sys
import numpy
import collections

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

