#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Wed 14 Oct 16:37:48 CEST 2015

import os, sys
import numpy

import bob.ip.draw
import bob.ip.color


def kp66_to_mask(image, keypoints, indent=10, plot=False):
  """kp66_to_mask(image, keypoints[, indent][, plot]) -> mask, mask_points
    
    This function builds a mask on the lower part of the face

    The mask is built using selected keypoints retrieved  by a 
    Discriminative Response Map Fitting (DRMF) algorithm. Note that
    the DRMF is not implemented here, and that the keypoints 
    are loaded from file (and are not provided in the package).

    Note also that this function is explicitly made for the
    keypoints set generated by the Matlab software downloaded
    from http://ibug.doc.ic.ac.uk/resources/drmf-matlab-code-cvpr-2013/
    
    If you decide to use another keypoint detector, you may need to 
    rewrite a function to build the mask from your keypoints. 

    **Parameters:**
      
      ``image`` (3d numpy array):
        The current frame.
      
      ``keypoints`` (2d numpy array 66x2): 
        the set of 66 keypoints retrieved by DRMF.

      ``indent`` ([Optional] int):
        The percentage of the facewidth [in pixels] by which 
        selected keypoints are shifted inside the face to
        build the mask. THe facewidth is defined by the distance 
        between the two keypoints located on the right and left edge
        of the face, at the eyes' height.
        Default to 10.
      
      ``plot`` ([Optional] boolean):
        If set to True, plots the current face with the
        selected keypoints and the built mask. Default to False

    **Returns**
      
      ``mask`` (2d numpy boolean array):
        A boolean array of the size of the original image, where the region
        corresponding to the mask is True.

      ``mask_points`` (list of tuples, 9x2):
        The points corresponding to vertices of the mask.
  """
  assert keypoints.shape[0] == 66, "You should provide a set 66 keypoints"

  if plot:
    imkey = numpy.copy(image)

  # there are 9 points to define the mask
  mask_points = []
  for k in range(keypoints.shape[0]):
    if k == 1 or k == 3 or k == 5 or k == 8 or k == 11 or k == 13 or k == 15 or k == 41 or k == 47:
      mask_points.append([int(keypoints[k, 0]), int(keypoints[k, 1])])
      if plot:
        bob.ip.draw.cross(imkey, (int(keypoints[k, 0]), int(keypoints[k, 1])), 4, (255,0,0))

  # indent in pixel will be the percentage provided of the face width
  # face width is defined as the distance between point 0 and point 6
  face_width = numpy.sqrt(numpy.sum((numpy.array(mask_points[0]) - numpy.array(mask_points[6]))**2))
  indent = int(float(indent)/100 * face_width)

  # left contour
  mask_points[0][1] += indent
  mask_points[1][1] += indent
  mask_points[2][1] += indent
  # chin tip
  mask_points[3][0] -= indent
  # right contour
  mask_points[4][1] -= indent
  mask_points[5][1] -= indent
  mask_points[6][1] -= indent
  # below left eye
  mask_points[7][0] += indent
  # below right eye
  mask_points[8][0] += indent

  # swap left and right eye such that vertices are following each other
  swp = mask_points[7]
  mask_points[7] = mask_points[8]
  mask_points[8] = swp

  if plot:
    for k in range(len(mask_points)-1):
      bob.ip.draw.line(imkey, (mask_points[k][0], mask_points[k][1]), (mask_points[k+1][0], mask_points[k+1][1]), (0,255,0))
    bob.ip.draw.line(imkey, (mask_points[0][0], mask_points[0][1]), (mask_points[8][0], mask_points[8][1]), (0,255,0))
    from matplotlib import pyplot
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(imkey, 2),2))
    pyplot.title('Built mask')
    pyplot.show()

  mask = get_mask(image, mask_points)
  return mask_points, mask


def get_mask(image, mask_points):
  """get_mask(image, mask_points) -> mask
    
  This function returns a boolean array where the mask is True.
  
  It turns mask points into a region of interest and returns the
  corresponding boolean array, of the same size as the image.
  Taken from https://github.com/jdoepfert/roipoly.py/blob/master/roipoly.py

  **Parameters**

      ``image`` (3d numpy array): 
        The current frame.

      ``mask_points`` (list of tuples, 9x2):
        The points corresponding to vertices of the mask.
  
  **Returns**

      ``mask`` (2d numpy boolean array):
        A boolean array of the size of the original image, where the region
        corresponding to the mask is True.
  """
  import matplotlib.path as mplPath

  ny = image.shape[1]
  nx = image.shape[2]
  poly_verts = [(mask_points[0][1], mask_points[0][0])]
  for i in range(len(mask_points)-1, -1, -1):
      poly_verts.append((mask_points[i][1], mask_points[i][0]))

  x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
  x, y = x.flatten(), y.flatten()
  points = numpy.vstack((x,y)).T

  ROIpath = mplPath.Path(poly_verts, closed=True)
  grid = ROIpath.contains_points(points).reshape((ny,nx))
  grid = grid.astype('bool')
  return grid


def  get_good_features_to_track(face, npoints, quality=0.01, min_distance=10, plot=False):
  """get_good_features_to_track(face, npoints[, quality][, min_distance][, plot]) -> corners
    
  This function applies the openCV function "good features to track"

  **Parameters**

    ``face`` (3d numpy array):
      The cropped face image

    ``npoints`` (int):
      The maximum number of strong corners you want to detect
  
    ``quality`` ([Optional] float):
      The minimum relative quality of the detected corners.
      Note that increasing this value decreases the number of
      detected corners. Defaluts to 0.01.

    ``min_distance`` ([Optional] int): 
      minimum euclidean distance between detected corners.
      Defaults to 10.
  
    ``plot`` ([Optional] boolean):
      if we should plot the currently selected features to track.
      Defaults to False.

  **Returns**

    ``corners`` (numpy array of dim (npoints, 1, 2)):
      the detected strong corners.
  """
  from cv2 import goodFeaturesToTrack
  gray = bob.ip.color.rgb_to_gray(face)
  #corners = cv2.goodFeaturesToTrack(gray, npoints, quality, min_distance)
  corners = goodFeaturesToTrack(gray, npoints, quality, min_distance)

  if plot:
    display = numpy.copy(face)
    int_corners = numpy.int0(corners)
    for i in int_corners:
      x,y = i.ravel()
      bob.ip.draw.cross(display, (y, x), 3, (255,0,0))
    from matplotlib import pyplot
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
    pyplot.title('Good features to track')
    pyplot.show()

  return corners

def track_features(previous, current, previous_points, plot=False):
  """track_features(previous, current, previous_points[, plot]) -> current points
  
  This function projects the features from the previous frame in the current frame.

  **Parameters**
  
    ``previous`` (3d numpy array):
      the previous frame.
  
    ``current`` (3d numpy array): 
      the current frame.
  
    ``previous_points`` (numpy array of dim (npoints, 1, 2)):
      the set of keypoints to track (in the previous frame).
  
    ``plot`` ([Optional] boolean):
      Plots the keypoints projected on the current frame.
      Defaults to False.

  **Returns**
  
    ``current_points`` (numpy array of dim (npoints, 1, 2)):
      the set of keypoints in the current frame.    
  """
  prev_gray = bob.ip.color.rgb_to_gray(previous)
  curr_gray = bob.ip.color.rgb_to_gray(current)
  from cv2 import calcOpticalFlowPyrLK
  current_points = calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPts=previous_points, nextPts=None)

  if plot:
    display = numpy.copy(current)
    int_corners = numpy.int0(current_points[0])
    for i in int_corners:
      x,y = i.ravel()
      bob.ip.draw.cross(display, (y, x), 3, (255,0,0))
    from matplotlib import pyplot
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
    pyplot.title('Result of the tracked features')
    pyplot.show()

  return current_points[0]


def find_transformation(previous_points, current_points):
  """find_transformation(previous_points, current_points) -> transformation matrix:
  
  This function finds the transformation matrix from previous points to current points.
  
  The transformation matrix is found using estimateRigidTransform 
  (fancier alternatives have been tried, but are not that stable).

  **Parameters**

    ``previous_points`` (numpy array):
      Set of 'starting' 2d points 

    ``current_points`` (numpy array):
      Set of 'destination' 2d points

  **Returns**
    
    ``transformation_matrix`` (numpy array of dim (3,2)):
      the affine transformation matrix between
      the two sets of points. 
  """
  from cv2 import estimateRigidTransform
  #return cv2.estimateRigidTransform(previous_points, current_points, False)
  return estimateRigidTransform(previous_points, current_points, False)


def get_current_mask_points(previous_mask_points, transfo_matrix):
  """get_current_mask_points(previous_mask_points, transfo_matrix) -> current_mask_points
  
  This projects the previous mask points to get the current mask.

  **Parameters**
  
    ``previous_mask_points`` (numpy array):
      The points forming the mask in the previous frame
  
    ``transformation_matrix`` (numpy array (3x2)):
      the affine transformation matrix between
      the two sets of points. 

  **Returns**
    
    ``current_mask_points`` (numpy array):
      The points forming the mask in the current frame
  """
  previous_mask_points = numpy.array([previous_mask_points], dtype='float64')
  from cv2 import transform
  #new_mask_points = cv2.transform(previous_mask_points, transfo_matrix)
  new_mask_points = transform(previous_mask_points, transfo_matrix)
  return new_mask_points[0].tolist()


def compute_average_colors_mask(image, mask, plot=False):
  """compute_average_colors_mask(image, mask[, plot]) ->  green_color
  
  This function computes the average green color within a given mask.

  **Parameters**
  
    ``image`` (3d numpy array ):
      The image containing the face.
    
    ``mask`` (2d numpy boolean array):
      A boolean array of the size of the original image, where the region
      corresponding to the mask is True.

    ``plot`` ([Optional] boolean):
      Plot the mask as an overlay on the original image.
      Defaults to False.
  
  **Returns**

    ``color`` (float):
      The average green color inside the mask ROI.
  """
  if plot:
    from matplotlib import pyplot
    display = numpy.copy(image)
    display[:, mask] = 255
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
    pyplot.title('Mask overlaid on the original frame')
    pyplot.show()

  green = image[1, mask]
  return numpy.mean(green)

def compute_average_colors_wholeface(image, plot=False):
  """compute_average_colors_mask(image [, plot]) ->  green_color
  
  This function computes the average green color within the provided face image 

  **Parameters**
  
    ``image`` (3d numpy array ):
      The cropped face image

    ``plot`` ([Optional] boolean):
      Plot the mask as an overlay on the original image.
      Defaults to False.
  
  **Returns**

    ``color`` (float):
      The average green color inside the face 
  """
  if plot:
    from matplotlib import pyplot
    display = numpy.copy(image)
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
    pyplot.title('Face area used to compute the mean green value')
    pyplot.show()

  green = image[1, :]
  return numpy.mean(green)
