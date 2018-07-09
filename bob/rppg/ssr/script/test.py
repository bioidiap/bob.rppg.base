#!/usr/bin/env python
# encoding: utf-8

import nose.tools
import pkgutil
import os, sys
import numpy
import functools

from bob.io.base.test_utils import datafile
from bob.io.base import load


def test_get_skin_pixels():
  """
  Test the skin colored pixels detection
  """

  # to run face detection
  import bob.ip.facedetect
  
  mod = sys.modules.get(__name__) or loader.load_module(__name__)
 
  # load face image
  face = load(datafile('001.jpg', 'bob.rppg.base'))

  from bob.rppg.ssr.ssr_utils import get_skin_pixels
  
  # zero threshold -> the number of skin pixels is the number of pixels in the cropped face
  skin_pixels = get_skin_pixels(face, 0, True, 0.0)
  bbox, quality = bob.ip.facedetect.detect_single_face(face)
  assert skin_pixels.shape[1] == (bbox.size[0] - 1) * bbox.size[1] # -1 because of the cropping

  # same as before, but with provided bbox
  bounding_boxes = [bbox] 
  skin_pixels = get_skin_pixels(face, 0, True, 0, bounding_boxes)
  assert skin_pixels.shape[1] == (bbox.size[0] - 1) * bbox.size[1] # -1 because of the cropping

  # threshold of 1.0 -> zero skin pixels
  skin_pixels = get_skin_pixels(face, 0, True, 1, bounding_boxes)
  assert skin_pixels.shape[1] == 0


def test_get_eigen():
  """
  Test the computation of eigenvalues and eigenvector
  """
  from bob.rppg.ssr.ssr_utils import get_eigen
  a = numpy.array([[0, 0], [0, 0]])
  evals, evecs = get_eigen(a) 
  assert numpy.all(evals == numpy.array([0, 0]))
  assert numpy.all(evecs == numpy.array([[0, 1], [1, 0]]))

