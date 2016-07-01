#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Fri  8 Apr 09:23:39 CEST 2016


"""Signals extractor for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...]
           [--dbdir=<path>] [--bboxdir=<path>] [--facedir=<path>] [--bgdir=<path>] 
           [--npoints=<int>] [--indent=<int>] [--quality=<float>] [--distance=<int>]
           [--overwrite] [--verbose ...] [--plot]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -d, --dbdir=<path>        The path to the database on your disk. If not set,
                            defaults to Idiap standard locations.
  -B, --bboxdir=<path>      The path to the faces bounding boxes (if any,
                            run face detection otherwise).
  -f, --facedir=<path>      The path to the directory where signal extracted 
                            from the face area will be stored [default: face]
  -b, --bgdir=<path>        The path to the directory where signal extracted 
                            from the background area will be stored [default: background]
  -n, --npoints=<int>       Number of good features to track [default: 40]
  -i, --indent=<int>        Indent (in percent of the face width) to apply to 
                            keypoints to get the mask [default: 10]
  -q, --quality=<float>     Quality level of the good features to track
                            [default: 0.01]
  -e, --distance=<int>      Minimum distance between detected good features to
                            track [default: 10]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.


Example:

  To run the signal extractor for the cohface database

    $ %(prog)s cohface -v

See '%(prog)s --help' for more information.

"""

import os
import sys
import pkg_resources

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("extract_log")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base
import bob.ip.facedetect

from ...base.utils import load_bbox
from ...base.utils import load_bbox_new
from ...base.utils import crop_face
from ...base.utils import crop_face_new

from ..extract_utils import kp66_to_mask
from ..extract_utils import get_good_features_to_track
from ..extract_utils import track_features
from ..extract_utils import find_transformation
from ..extract_utils import get_current_mask_points
from ..extract_utils import get_mask 
from ..extract_utils import compute_average_colors_mask

# TODO: This is not really necessary, only needed
# to be compliant with Andre's stuff - Guillaume HEUSCH, 11-04-2016
import collections
Point = collections.namedtuple('Point', 'y,x')
BoundingBox = collections.namedtuple('BoundingBox', 'topleft,size,quality')

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(
          prog=prog,
          version=version,
          )
  args = docopt(
      __doc__ % completions,
      argv=arguments,
      version='Signal extractor for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger("extract_log").setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger("extract_log").setLevel(logging.DEBUG)

  # chooses the database driver to use
  if args['cohface']:
    import bob.db.cohface
    if os.path.isdir(bob.db.cohface.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.cohface.DATABASE_LOCATION
    elif args['--indir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--indir']
    else:
      logger.warn("Could not find the database directory, please provide one")
      sys.exit()
    db = bob.db.cohface.Database(dbdir)
    if not((args['--protocol'] == 'all') or (args['--protocol'] == 'clean') or (args['--protocol'] == 'natural')):
      logger.warning("Protocol should be either 'clean', 'natural' or 'all' (and not {0})".format(args['--protocol']))
      sys.exit()
    objects = db.objects(args['--protocol'], args['--subset'])

  elif args['hci']:
    import bob.db.hci_tagging
    import bob.db.hci_tagging.driver
    if os.path.isdir(bob.db.hci_tagging.driver.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.hci_tagging.driver.DATABASE_LOCATION
    elif args['--indir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--indir'] 
    else:
      logger.warn("Could not find the database directory, please provide one")
      sys.exit()
    db = bob.db.hci_tagging.Database()
    if not((args['--protocol'] == 'all') or (args['--protocol'] == 'cvpr14')):
      logger.warning("Protocol should be either 'all' or 'cvpr14' (and not {0})".format(args['--protocol']))
      sys.exit()
    objects = db.objects(args['--protocol'], args['--subset'])

  # if we are on a grid environment, just find what I have to process.
  if os.environ.has_key('SGE_TASK_ID'):
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(objects):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(objects))
    objects = [objects[pos]]

  # does the actual work - for every video in the available dataset, 
  # extract the signals and dumps the results to the corresponding directory
  for obj in objects:

    # expected output file
    output_face = obj.make_path(args['--facedir'], '.hdf5')
    output_bg = obj.make_path(args['--bgdir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if (os.path.exists(output_face) and os.path.exists(output_bg)) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output_face)
      continue
    
    # load video
    video = obj.load_video(dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # load the result of face detection, if provided
    # face detection will be run otherwise
    if bool(args['--bboxdir']):
      bbox_file = obj.make_path(args['--bboxdir'], '.face')
      logger.debug("Loading bounding boxes")
      try:
        #bounding_boxes = load_bbox(bbox_file)
        bounding_boxes = load_bbox_new(bbox_file)
      except IOError as e:
        logger.warn("Detecting faces in file `%s' (no bounding box file available)", obj.stem)
    else:
      logger.warn("Detecting faces in file `%s' (no bounding box file available)", obj.stem)

    # average green color in the mask area  
    face_color = numpy.zeros(len(video), dtype='float64')
    # average green color in the background area
    bg_color = numpy.zeros(len(video), dtype='float64')

    # loop on video frames
    for i, frame in enumerate(video):
      logger.debug("Processing frame %d/%d...", i+1, len(video))

      if i == 0:
        # first frame:
        # -> load the keypoints detected by DMRF
        # -> infer the mask from the keypoints
        # -> detect the face
        # -> get "good features" inside the face
        kpts = obj.load_drmf_keypoints()
        mask_points, mask = kp66_to_mask(frame, kpts, int(args['--indent']), bool(args['--plot']))

        try: 
          bbox = bounding_boxes[i]
        except NameError:
          bbox, quality = bob.ip.facedetect.detect_single_face(frame)
          # TODO: should be removed, use only the result
          # of detect_single_face - Guillaume HEUSCH, 11-04-2016
          #bbox = BoundingBox(Point(*bb.topleft), Point(*bb.size), quality)
        
        print bbox.size

        # define the face width for the whole sequence
        facewidth = bbox.size[1]
        #face = crop_face(frame, bbox, int(args['--facewidth']))
        face = crop_face_new(frame, bbox, facewidth)
        good_features = get_good_features_to_track(face,int(args['--npoints']), 
            float(args['--quality']), int(args['--distance']), bool(args['--plot']))
      else:
        # subsequent frames:
        # -> crop the face with the bounding_boxes of the previous frame (so
        #    that faces are of the same size)
        # -> get the projection of the corners detected in the previous frame
        # -> find the (affine) transformation relating previous corners with
        #    current corners
        # -> apply this transformation to the mask
        #face = crop_face(frame, prev_bb, int(args['--facewidth']))
        face = crop_face_new(frame, prev_bb, facewidth)
        good_features = track_features(prev_face, face, prev_features,
            bool(args['--plot']))
        project = find_transformation(prev_features, good_features)
        if project is None: 
          logger.warn("Sequence {0}, frame {1} : No projection was found"
              " between previous and current frame, mask from previous frame will be used"
              .format(obj.stem, i))
        else:
          mask_points = get_current_mask_points(mask_points, project)

      # update stuff for the next frame:
      # -> the previous face is the face in this frame, with its bbox (and not
      #    with the previous one)
      # -> the features to be tracked on the next frame are re-detected
      try: 
        prev_bb = bounding_boxes[i]
      except NameError:
        bb, quality = bob.ip.facedetect.detect_single_face(frame)
        # TODO: should be removed, use only the result
        # of detect_single_face - Guillaume HEUSCH, 11-04-2016
        #prev_bb = BoundingBox(Point(*bb.topleft), Point(*bb.size), quality)
        prev_bb = bb

      #prev_face = crop_face(frame, prev_bb, int(args['--facewidth']))
      prev_face = crop_face_new(frame, prev_bb, facewidth)
      prev_features = get_good_features_to_track(face, int(args['--npoints']),
          float(args['--quality']), int(args['--distance']),
          bool(args['--plot']))
      if prev_features is None:
        logger.warn("Sequence {0}, frame {1} No features to track"  
            " detected in the current frame, using the previous ones"
            .format(obj.stem, i))
        prev_features = good_features

      # get the bottom face region average colors
      face_mask = get_mask(frame, mask_points)
      face_color[i] = compute_average_colors_mask(frame, face_mask, bool(args['--plot']))

      # get the background region average colors
      bg_mask = numpy.zeros((frame.shape[1], frame.shape[2]), dtype=bool)
      bg_mask[:100, :100] = True
      bg_color[i] = compute_average_colors_mask(frame, bg_mask, bool(args['--plot']))

    # saves the data into an HDF5 file with a '.hdf5' extension
    out_facedir = os.path.dirname(output_face)
    if not os.path.exists(out_facedir): bob.io.base.create_directories_safe(out_facedir)
    bob.io.base.save(face_color, output_face)
    logger.info("Output file saved to `%s'...", output_face)

    out_bgdir = os.path.dirname(output_bg)
    if not os.path.exists(out_bgdir): bob.io.base.create_directories_safe(out_bgdir)
    bob.io.base.save(bg_color, output_bg)
    logger.info("Output file saved to `%s'...", output_bg)
