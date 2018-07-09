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

""" Test Units
"""

import nose.tools
import pkgutil
import os, sys
import numpy
import functools

def test_compute_mean_rgb():
  """
  Test the mean color computation inside a pre-defined area
  """
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  mask = numpy.zeros((100, 100), dtype='bool')
  mask[20:80, 20:80] = True
  image[0, :, :] = 0
  image[1, :, :] = 128
  image[2, :, :] = 255
  image[0, 0, 0] = 255
  image[1, 0, 0] = 0
  image[2, 0, 0] = 0
  
  from bob.rppg.chrom.extract_utils import compute_mean_rgb
  r, g, b = compute_mean_rgb(image, mask)
  assert r == 0
  assert g == 128 
  assert b == 255

def test_compute_gray_diff():
  """
  Test the computation of the grayscale difference beween two images
  """
  image1 = numpy.zeros((3, 100, 100), dtype='uint8')
  image2 = numpy.ones((3, 100, 100), dtype='uint8')*255

  from bob.rppg.chrom.extract_utils import compute_gray_diff
  diff = compute_gray_diff(image1, image1)
  assert diff == 0.0
  diff = compute_gray_diff(image2, image2)
  assert diff == 0.0
  diff = compute_gray_diff(image1, image2)
  assert diff == 100*100 

def test_select_stable_frames():
  """
  Test the selection of stable consecutive frames
  """
  from bob.rppg.chrom.extract_utils import select_stable_frames
  diff = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  idx = select_stable_frames(diff, 10)
  assert idx == 0
  diff = numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
  idx = select_stable_frames(diff, 5)
  assert idx == 3 
  idx = select_stable_frames(diff, 10)
  assert idx == 0


def test_project_chrominance():
  """
  Tests the colorspace conversion
  """
  from bob.rppg.chrom.extract_utils import project_chrominance
  r = g = b = 0
  x,y = project_chrominance(r, g, b)
  assert x == 0
  assert y == 0
  r = g = b = 1
  x,y = project_chrominance(r, g, b)
  assert x == 1.0
  assert y == 1.0
