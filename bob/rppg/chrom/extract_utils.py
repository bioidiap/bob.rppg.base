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

import numpy
import bob.ip.base

def compute_mean_rgb(image, mask=None):
  """compute_mean_rgb(image, mask=None) -> mean_r, mean_g, mean_b
  This function computes the mean R, G and B of an image.
  
  Note that a mask could be provided to tell which pixels should
  be taken into account when computing the mean.
  
  **Parameters**

    ``image`` (3d numpy array):
      The image to process

    ``mask`` (2d boolen numpy array):
      Mask of the size of the image, telling which pixels
      should be considered

  **Returns**

    ``mean_r`` (float):
      The mean red value

    ``mean_g`` (float):
      The mean green value

    ``mean_b`` (float):
      The mean blue value
  """
  assert len(image.shape) == 3, "This is meant to work with color images (3 channels)"
  mean_r = numpy.mean(image[0, mask])
  mean_g = numpy.mean(image[1, mask])
  mean_b = numpy.mean(image[2, mask])
  return mean_r, mean_g, mean_b


def compute_gray_diff(previous, current):
  """compute_gray_diff(previous, current) -> diff 
  
  This function computes the difference in intensity between two images .
  
  **Parameters**

    ``previous`` (3d numpy array):
      The previous frame.
 
    ``current`` (3d numpy array):
      The current frame.
 
  **Returns**

    ``diff`` (float):
      The sum of the absolute difference in pixel intensity between two frames
  """
  from bob.ip.color import rgb_to_gray
  prevg = rgb_to_gray(previous)
  currg = rgb_to_gray(current)
  return numpy.sum(numpy.absolute(prevg - currg))


def select_stable_frames(diff, n):

  """select_stable_frames(diff, n) -> index

  This functions selects a stable subset of consecutive frames
 
  The selection is made by considering the grayscale difference between frames.
  The subset is chosen as the one for which the sum of difference is minimized

  **Parameters**

    ``diff`` (1d numpy array):
      The sum of absolute pixel intensity differences between 
      consecutive frames, across the whole sequence.

    ``n`` ; (int):
      The number of consecutive frames you want to select.

  **Returns**

    ``index`` (int):
      The frame index at which the stable segment begins.
  """
  current_min = float("inf")
  current_index = 0
  for i in range(0, diff.shape[0]-n, 1):
    current_sum = sum(diff[i: i+n])
    if current_sum < current_min:
      current_index = i
      current_min = current_sum
  return current_index


def project_chrominance(r, g, b):
  """
  Projects rgb values onto the x and y chrominance space
  
  See equation (9) of [dehaan-tbe-2013]_.

  **Parameters**

    ``r`` (float):
      The red value

    ``g`` (float):
      The green value

    ``b`` (float):
      The blue value

  **Returns**
  
    ``x`` (float):
      The x value
  
    ``y`` (float):
      The y value
  """
  x = (3.0 * r) - (2.0 * g)
  y = (1.5 * r) + g + (1.5 * b)
  return x, y
