#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Tue 26 Apr 17:19:09 CEST 2016

import collections
import numpy
import bob.ip.base

Point = collections.namedtuple('Point', 'y,x')
BoundingBox = collections.namedtuple('BoundingBox', 'topleft,size,quality')

from bob.ip.skincolorfilter import SkinColorFilter
skin_filter = SkinColorFilter()

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

def get_skin_pixels(face_frame, index, facewidth, skininit, threshold, bounding_boxes=None, skin_frame=None, plot=False):
  """get_skin_pixels(face_frame, index, facewidth, skininit, threshold, bounding_boxes=None, skin_frame=None, plot=False) -> skin_pixels
 
    Get a list of skin colored pixels inside the given frame.
    
    **Parameters**

    ``face_frame`` : (numpy array)
      The frame where the face has to be detected.

    ``index`` : (int)
      The index of the frame containing the face to be detected.

    ``facewidth`` : (int)
      The width of the cropped face

    ``skininit`` : (boolean)
      Flag if you want the parameters of the skin model to be re-estimated.

    ``threshold`` : (float [0-1])
      The threshold on the skin color probability.

    ``bounding_boxes`` : (list of BoundingBoxes)
      The face bounding boxes corresponding to the sequence.

    ``skin_frame`` : (numpy array)
      The frame where the skin pixels have to be retrieved.
      If not set, face_frame will be used.

    ``plot`` : (boolean)
      Flag to plot the result of skin pixels detection

    **Returns**

      ``skin_pixels`` : (numpy array)
        The RGB values of all detected skin colored pixels
  """
  if skin_frame is None:
    skin_frame = face_frame

  if bounding_boxes: 
    bbox = bounding_boxes[index]
  else:
    bb, quality = bob.ip.facedetect.detect_single_face(face_frame)
    # TODO: should be removed, use only the result
    # of detect_single_face - Guillaume HEUSCH, 11-04-2016
    bbox = BoundingBox(Point(*bb.topleft), Point(*bb.size), quality)

  face = crop_face(skin_frame, bbox, facewidth)

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

      ``skin_pixels`` : (numpy array)
        The RGB values of skin-colored pixels.
        
    **Returns**

      ``eigenvalues`` : (numpy array)
        The eigenvalues of the correlation matrix

      ``eigenvectors`` : (numpy array)
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

      ``skin_pixels`` : (numpy array)
        The RGB values of skin-colored pixels.

      ``eigenvectors`` : (numpy array)
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
  
      ``counter`` : (int)
        The frame index

      ``temporal_stride`` : (int)
        The temporal stride to use

      ``eigenvectors`` : (numpy array)
        The eigenvectors of the c matrix (for all frames up to counter). 
    
      ``eigenvalues`` : (numpy array)
        The eigenvalues of the c matrix (for all frames up to counter).

      ``plot`` : (boolean)
        If you want something to be plotted

    **Returns**

      ``p`` : (numpy array)
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