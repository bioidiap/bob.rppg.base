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
from ..base.utils import crop_face

from bob.ip.skincolorfilter import SkinColorFilter
skin_filter = SkinColorFilter()

def get_skin_pixels(face_frame, index, skininit, threshold, bounding_boxes=None, skin_frame=None, plot=False):
  """get_skin_pixels(face_frame, index, facewidth, skininit, threshold, bounding_boxes=None, skin_frame=None, plot=False) -> skin_pixels
 
    Get a list of skin colored pixels inside the given frame.
    
    **Parameters**

    ``face_frame`` (numpy array):
      The frame where the face has to be detected.

    ``index`` (int):
      The index of the frame containing the face to be detected.

    ``skininit`` (boolean):
      Flag if you want the parameters of the skin model to be re-estimated.

    ``threshold`` (float [0-1]):
      The threshold on the skin color probability.

    ``bounding_boxes`` (list of BoundingBoxes):
      The face bounding boxes corresponding to the sequence.

    ``skin_frame`` (numpy array):
      The frame where the skin pixels have to be retrieved.
      If not set, face_frame will be used.

    ``plot`` (boolean):
      Flag to plot the result of skin pixels detection

    **Returns**

      ``skin_pixels`` (numpy array):
        The RGB values of all detected skin colored pixels
  """
  if skin_frame is None:
    skin_frame = face_frame

  if bounding_boxes: 
    bbox = bounding_boxes[index]
  else:
    bbox, quality = bob.ip.facedetect.detect_single_face(face_frame)

  face = crop_face(skin_frame, bbox, bbox.size[1])

  if skininit:
    skin_filter.estimate_gaussian_parameters(face)
  skin_mask = skin_filter.get_skin_mask(face, threshold)
  skin_pixels = face[:, skin_mask]

  if plot:
    from matplotlib import pyplot
    skin_mask_image = numpy.copy(face)
    skin_mask_image[:, skin_mask] = 255
    pyplot.title("skin pixels in frame {0}".format(index))
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
    pyplot.show()
  
  skin_pixels = skin_pixels.astype('float64') / 255.0
  return skin_pixels

def get_eigen(skin_pixels):
  """get_eigen(skin_pixels) -> eigenvalues, eigenvectors
  
    Build the C matrix, get eigenvalues and eigenvectors, sort them.

    **Parameters**

      ``skin_pixels`` (numpy array):
        The RGB values of skin-colored pixels.
        
    **Returns**

      ``eigenvalues`` (numpy array):
        The eigenvalues of the correlation matrix

      ``eigenvectors`` (numpy array):
        The (sorted) eigenvectors of the correlation matrix

  """
  # build the correlation matrix
  c = numpy.dot(skin_pixels, skin_pixels.T)
  c /= skin_pixels.shape[1]

  # get eigenvectors and sort them according to eigenvalues (largest first)
  evals, evecs = numpy.linalg.eig(c)
  idx = evals.argsort()[::-1]   
  eigenvalues = evals[idx]
  eigenvectors = evecs[:,idx]
  return eigenvalues, eigenvectors

def plot_eigenvectors(skin_pixels, eigenvectors):
  """plot_eigenvectors(skin_pixels, eigenvectors, counter)
  
    Plots skin pixel cluster and eignevectors in the RGB space.

    **Parameters**

      ``skin_pixels`` (numpy array):
        The RGB values of skin-colored pixels.

      ``eigenvectors`` (numpy array):
        The eigenvectors of the correlation matrix. 
  """
  origin = numpy.array([0, 0, 0])
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(skin_pixels[0], skin_pixels[1], skin_pixels[2])
  ax.plot([origin[0], eigenvectors[0, 0]], [origin[1], eigenvectors[1, 0]], zs=[origin[2], eigenvectors[2, 0]], color='g')
  for k in range(1,3,1):
    ax.plot([origin[0], eigenvectors[k, 0]], [origin[1], eigenvectors[k, 1]], zs=[origin[2], eigenvectors[k, 2]], color='r')
  plt.show()

def build_P(counter, temporal_stride, eigenvectors, eigenvalues, plot=False):
  """build_P(counter, temporal_stride, eigenvectors, eigenvalues, plot=False) -> p
  
    Builds P

    **Parameters**
  
      ``counter`` (int):
        The frame index

      ``temporal_stride`` (int):
        The temporal stride to use

      ``eigenvectors`` (numpy array):
        The eigenvectors of the c matrix (for all frames up to counter). 
    
      ``eigenvalues`` (numpy array):
        The eigenvalues of the c matrix (for all frames up to counter).

      ``plot`` (boolean):
        If you want something to be plotted

    **Returns**

      ``p`` (numpy array):
        The p signal to add to the pulse.
  """
  tau = counter - temporal_stride
         
  # SR'
  sr_prime_vec = numpy.zeros((3, temporal_stride), 'float64')
  c2 = 0
  for t in range(tau, counter, 1):
    # equation 11
    sr_prime = numpy.sqrt(eigenvalues[0, t] / eigenvalues[1, tau]) * numpy.dot(eigenvectors[:, 0, t].T, numpy.outer(eigenvectors[:, 1, tau], eigenvectors[:, 1, tau].T))
    sr_prime += numpy.sqrt(eigenvalues[0, t] / eigenvalues[2, tau]) * numpy.dot(eigenvectors[:, 0, t].T, numpy.outer(eigenvectors[:, 2, tau], eigenvectors[:, 2, tau].T))
    sr_prime_vec[:, c2] = sr_prime
    c2 += 1
  
  # build p and add it to the final pulse signal (equation 12 and 13)
  p = sr_prime_vec[0, :] - ((numpy.std(sr_prime_vec[0, :])/numpy.std(sr_prime_vec[1, :])) * sr_prime_vec[1, :])
 
  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(temporal_stride), sr_prime_vec[0, :], c='b')
    ax.plot(range(temporal_stride), sr_prime_vec[1, :], c='b')
    ax.plot(range(temporal_stride), p, c='r')
    plt.show()
  
  return p
