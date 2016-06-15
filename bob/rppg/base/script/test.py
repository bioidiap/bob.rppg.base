import os, sys
import numpy

def test_load_bbx():
  """
  Test the loading of a bounding box file
  """

  # to get the filename relative to the package ...
  resource = 'data/bbox.face'
  mod = sys.modules.get(__name__) or loader.load_module(__name__)
  parts = resource.split('/')
  parts.insert(0, os.path.dirname(mod.__file__))
  resource_name = os.path.join(*parts)

  from bob.rppg.base.utils import load_bbox
  bounding_boxes = load_bbox(resource_name)

  assert len(bounding_boxes) == 2410
  bbox = bounding_boxes[0]
  assert bbox.topleft.x == 283
  assert bbox.topleft.y == 143
  assert bbox.size.x == 125 
  assert bbox.size.y == 150
  bbox = bounding_boxes[2049]
  assert bbox.topleft.x == 265
  assert bbox.topleft.y == 154
  assert bbox.size.x == 120
  assert bbox.size.y == 144

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
  # XXX a bug has been identified, will be corrected once the 
  # code has been released

  # XXX also, this should be modified to use bob.ip.facedetect classes
  import collections
  Point = collections.namedtuple('Point', 'y,x')
  BoundingBox = collections.namedtuple('BoundingBox', 'topleft,size,quality')

  image = numpy.zeros((3, 100, 100), dtype='uint8')
  bbox = BoundingBox(Point(20, 20), Point(50, 50), 0.99)

  from bob.rppg.base.utils import crop_face
  cropped = crop_face(image, bbox, 48)
  assert cropped.shape == (3, 48, 48)

