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

def test_kp66_to_mask():
  """
  Test the mask constructions based on DRMF keypoints
  """
  keypoints = numpy.zeros((66,2))

  # dummy keypoints that are considered to build the mask
  keypoints[1] = numpy.array([10, 10])
  keypoints[3] = numpy.array([20, 10])
  keypoints[5] = numpy.array([30, 10])
  keypoints[8] = numpy.array([50, 35])
  keypoints[11] = numpy.array([30, 60])
  keypoints[13] = numpy.array([20, 60])
  keypoints[15] = numpy.array([10, 60])
  keypoints[41] = numpy.array([10, 30])
  keypoints[47] = numpy.array([10, 40])

  # dummy image
  image = numpy.zeros((3, 100, 100), dtype='uint8')

  from bob.rppg.cvpr14.extract_utils import kp66_to_mask
  mask_points, mask = kp66_to_mask(image, keypoints, 10, False)
  assert numpy.array_equal(mask_points[0], numpy.array([10,15]))
  assert numpy.array_equal(mask_points[1], numpy.array([20,15]))
  assert numpy.array_equal(mask_points[2], numpy.array([30,15]))
  assert numpy.array_equal(mask_points[3], numpy.array([45,35]))
  assert numpy.array_equal(mask_points[4], numpy.array([30,55]))
  assert numpy.array_equal(mask_points[5], numpy.array([20,55]))
  assert numpy.array_equal(mask_points[6], numpy.array([10,55]))
  assert numpy.array_equal(mask_points[7], numpy.array([15,40]))
  assert numpy.array_equal(mask_points[8], numpy.array([15,30]))

  # check points that should be inside or outside the mask
  assert not mask[0,0]
  assert not mask[9,15]
  assert not mask[99, 99]
  assert mask[20, 16]
  assert mask[30, 30]
  assert mask[16, 40]
  

def opencv_available(test):
  """Decorator for detecting if OpenCV/Python bindings are available"""
  from nose.plugins.skip import SkipTest

  @functools.wraps(test)
  def wrapper(*args, **kwargs):
    try:
      import cv2
      return test(*args, **kwargs)
    except ImportError:
      raise SkipTest("The cv2 module is not available")

  return wrapper


@opencv_available
def test_gftt():
  """
  Tests the good features to track
  """
  # white square on a black background
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  image[:, 20:80, 20:80] = 255
  
  from bob.rppg.cvpr14.extract_utils import get_good_features_to_track 
  corners = get_good_features_to_track(image, 4)
  assert numpy.array_equal(corners[0][0], numpy.array([79.0,79.0])), "1st corner"
  assert numpy.array_equal(corners[1][0], numpy.array([20.0,79.0])), "2nd corner"
  assert numpy.array_equal(corners[2][0], numpy.array([79.0,20.0])), "3rd corner"
  assert numpy.array_equal(corners[3][0], numpy.array([20.0,20.0])), "4th corner"


@opencv_available
def test_track_features():
  """
  Tests the track features functions
  """

  # white square on a black background
  image1 = numpy.zeros((3, 100, 100), dtype='uint8')
  image1[:, 20:80, 20:80] = 255
  
  # white square on a black background - shifted by one pixel
  image2 = numpy.zeros((3, 100, 100), dtype='uint8')
  image2[:, 21:81, 21:81] = 255
  from bob.rppg.cvpr14.extract_utils import get_good_features_to_track 
  points1 = get_good_features_to_track(image1, 4)
  from bob.rppg.cvpr14.extract_utils import track_features
  points2 = track_features(image1, image2, points1)
  points2 = numpy.rint(points2)
  assert numpy.array_equal(points2[0][0], numpy.array([80,80])), "1st corner"
  assert numpy.array_equal(points2[1][0], numpy.array([21,80])), "2nd corner"
  assert numpy.array_equal(points2[2][0], numpy.array([80,21])), "3rd corner"
  assert numpy.array_equal(points2[3][0], numpy.array([21,21])), "4th corner"
  

@opencv_available
def test_find_transformation():
  """
  Test the function to find the homographic transformation
  """
  points1 = numpy.zeros((8,1, 2), dtype='int')
  points1[1, 0, :] = [1, 1] 
  points1[2, 0, :] = [0, 1] 
  points1[3, 0, :] = [1, 0]
  points1[4, 0, :] = [2, 2]
  points1[5, 0, :] = [2, 3]
  points1[6, 0, :] = [4, 3]
  points1[7, 0, :] = [5, 3]
  points2 = numpy.copy(points1)

  from bob.rppg.cvpr14.extract_utils import find_transformation
  mat = find_transformation(points1, points2)

  assert numpy.abs(mat[0, 0] - 1.0) < 1e-14   
  assert numpy.abs(mat[1, 1] - 1.0) < 1e-14   
  assert numpy.abs(mat[0, 1]) < 1e-14   
  assert numpy.abs(mat[1, 0]) < 1e-14   
  assert numpy.abs(mat[0, 2]) < 1e-14   
  assert numpy.abs(mat[1, 2]) < 1e-14   


def test_compute_average_color():
  """
  Test the mean color computation inside a pre-defined area
  """
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  mask = numpy.zeros((100, 100), dtype='bool')
  mask[20:80, 20:80] = True
  image[1, :, :] = 128
  
  from bob.rppg.cvpr14.extract_utils import compute_average_colors_mask
  mean_green = compute_average_colors_mask(image, mask)[1]
  assert mean_green == 128

def test_rectify_illumination():
  """
  Test the illumination rectification
  """
  signal = numpy.ones(100)
  target = numpy.ones(100)
  from bob.rppg.cvpr14.illum_utils import rectify_illumination

  # signal and target are equal -> output is zero
  output = rectify_illumination(signal, target, 1, 1)
  assert numpy.array_equal(output, numpy.zeros(100))


def test_build_segments():
  """
  Test the build segment function
  """
  signal = numpy.zeros(100)
  length = 10
  
  from bob.rppg.cvpr14.motion_utils import build_segments
  segments, end_index = build_segments(signal, length)
  assert segments.shape == (10, 10)
  assert end_index == 100

  length = 11
  segments, end_index = build_segments(signal, length)
  assert segments.shape == (9, 11)
  assert end_index == 99


def test_prune_segments():
  """
  Test the pruning of segments
  """
  segments = numpy.random.randn(10, 10)
  segments[0] = numpy.random.randn(10) * 10.0
  segments[4] = numpy.random.randn(10) * 10.0
  
  from bob.rppg.cvpr14.motion_utils import prune_segments
  pruned, gaps, cut_index = prune_segments(segments, 2.0) 

  # segments with high std should have been pruned
  assert pruned.shape == (8,10)
  
  # the first segment has been pruned, no gap should be accounted for
  assert not gaps[0]

  # the 5th segment has been pruned, a gap should be accounted for
  assert gaps[3]

  # two segments have been pruned
  assert len(cut_index) == 2
  # the first one
  assert cut_index[0] == (0, 10)
  # the fifth one
  assert cut_index[1] == (40, 50)


def test_build_final_signal():
  """
  Test the building of the final signal
  """
  segments = numpy.ones((10, 10))
  segments[4:] += 4
  gaps = [False] * 10
  gaps[4] = True

  from bob.rppg.cvpr14.motion_utils import build_final_signal
  signal = build_final_signal(segments, gaps)
  
  assert signal.shape[0] == 100
  assert numpy.array_equal(signal, numpy.ones(100))

def test_detrend():
  """
  Test the detrend filter
  """
  x = numpy.array(range(20))
  y = 2 + x
 
  # detrend of the signal
  # result should be more or less zero-mean and flat
  from bob.rppg.cvpr14.filter_utils import detrend
  filtered = detrend(y, 300)
  assert numpy.all(filtered < 1e-10)

def test_average():
  """
  Test the average filter
  """
  signal = numpy.random.randn(100)
  from bob.rppg.cvpr14.filter_utils import average
  filtered = average(signal, 1)
  # if the window is one, the signal should be unaltered
  assert numpy.array_equal(signal, filtered)
  
  signal = numpy.ones(100)
  filtered = average(signal, 5)
  # the signal is constant, so should be the result
  # after the window size has been reached
  assert filtered[0] == signal[0] / 5.0
  assert numpy.all(signal[5:] - filtered[5:] < 1e-15)
  filtered = average(signal, 17)
  assert filtered[0] == signal[0] / 17.0 
  assert numpy.all(signal[17:] - filtered[17:] < 1e-15)
