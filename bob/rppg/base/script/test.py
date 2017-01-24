import os, sys
import numpy

def test_scale_image():
  """
  Tests the rescaling of an image
  """

  from bob.rppg.base.utils import scale_image
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  scaled = scale_image(image, 50, 75)
  assert scaled.shape == (3, 50, 75)

def test_crop_face():
  """
  Test the cropping of a face
  """
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  
  from bob.ip.facedetect import BoundingBox
  bbox = BoundingBox((20, 20), (50, 50))

  from bob.rppg.base.utils import crop_face
  cropped = crop_face(image, bbox, 48)
  assert cropped.shape == (3, 48, 48)

