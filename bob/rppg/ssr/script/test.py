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

import nose.tools
import pkgutil
import os, sys
import numpy
import functools


def test_get_skin_pixels():
  """
  Test the skin colored pixels detection
  """

  # to run face detection
  import bob.ip.facedetect
  
  mod = sys.modules.get(__name__) or loader.load_module(__name__)
 
  # load face image
  face_file = 'data/001.jpg'
  parts = face_file.split('/')
  parts.insert(0, os.path.dirname(mod.__file__))
  face_name = os.path.join(*parts)
  import bob.io.base
  import bob.io.image
  face = bob.io.base.load(face_name)

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
  a = numpy.array([[1, 0], [0, 1]])

  from bob.rppg.ssr.ssr_utils import get_eigen
  evals, evecs = get_eigen(a) 
  assert numpy.all(evals == numpy.array([0, 0]))
  assert numpy.all(evecs == numpy.array([[0, 1], [1, 0]]))
  
  a = numpy.array([[0, 0], [0, 0]])
  evals, evecs = get_eigen(a) 
  assert numpy.all(evals == numpy.array([0, 0]))
  assert numpy.all(evecs == numpy.array([[0, 1], [1, 0]]))

