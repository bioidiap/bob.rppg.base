#!/usr/bin/env python
# encoding: utf-8

"""Extract pulses according to Li's CVPR 2014 algorithm 

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]
           [--facedir=<path>] [--bgdir=<path>] 
           [--npoints=<int>] [--indent=<int>] [--quality=<float>] [--distance=<int>]
           [--wholeface] [--overwrite] [--verbose ...] [--plot] [--gridcount]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the sets will be loaded.
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
  -g, --gridcount           Prints the number of objects to process and exits.
  -w, --wholeface           Consider the whole face region instead of the mask.


Example:

  To run the pulse extractor: 

    $ %(prog)s config.py -v

See '%(prog)s --help' for more information.

"""

from __future__ import print_function

import os
import sys
import pkg_resources

from bob.core.log import setup
logger = setup("bob.rppg.base")

from docopt import docopt

from bob.extension.config import load
from ...base.utils import get_parameter

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base
import bob.ip.facedetect

from ...base.utils import crop_face

from ..extract_utils import kp66_to_mask
from ..extract_utils import get_good_features_to_track
from ..extract_utils import track_features
from ..extract_utils import find_transformation
from ..extract_utils import get_current_mask_points
from ..extract_utils import get_mask 
from ..extract_utils import compute_average_colors_mask
from ..extract_utils import compute_average_colors_wholeface

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Signal extractor for videos (%s)' % version,)

  # load configuration file
  configuration = load([os.path.join(args['<configuration>'])])
 
  # get various parameters, either from config file or command-line 
  protocol = get_parameter(args, configuration, 'protocol', 'all')
  subset = get_parameter(args, configuration, 'subset', None)
  facedir = get_parameter(args, configuration, 'facedir', 'face')
  bgdir = get_parameter(args, configuration, 'bgdir', 'bg')
  npoints = get_parameter(args, configuration, 'npoints', 40)
  indent = get_parameter(args, configuration, 'indent', 10)
  quality = get_parameter(args, configuration, 'quality', 0.01)
  distance = get_parameter(args, configuration, 'distance', 10)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  gridcount = get_parameter(args, configuration, 'gridcount', False)
  wholeface = get_parameter(args, configuration, 'wholeface', False)
  verbosity_level = get_parameter(args, configuration, 'verbose', 0)

  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, verbosity_level)
  
  # TODO: find a way to check protocol names - Guillaume HEUSCH, 22-06-2018
  if hasattr(configuration, 'database'):
    objects = configuration.database.objects(protocol, subset)
  else:
    logger.error("Please provide a database in your configuration file !")
    sys.exit()

  # if we are on a grid environment, just find what I have to process.
  sge = False
  try:
    sge = os.environ.has_key('SGE_TASK_ID') # python2
  except AttributeError:
    sge = 'SGE_TASK_ID' in os.environ # python3
    
  if sge:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(objects):
      raise RuntimeError("Grid request for job {} on a setup with {} jobs".format(pos, len(objects)))
    objects = [objects[pos]]

  if gridcount:
    print(len(objects))
    sys.exit()

  # does the actual work - for every video in the available dataset, 
  # extract the signals and dumps the results to the corresponding directory
  for obj in objects:

    # expected output file
    output_face = obj.make_path(facedir, '.hdf5')
    output_bg = obj.make_path(bgdir, '.hdf5')

    # if output exists and not overwriting, skip this file
    if (os.path.exists(output_face) and os.path.exists(output_bg)) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output_face)
      continue
    
    # load video
    video = obj.load_video(configuration.dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # load the result of face detection
    bounding_boxes = obj.load_face_detection() 

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
        if not wholeface:
          kpts = obj.load_drmf_keypoints()
          mask_points, mask = kp66_to_mask(frame, kpts, indent, plot)

        try: 
          bbox = bounding_boxes[i]
        except NameError:
          bbox, quality = bob.ip.facedetect.detect_single_face(frame)
        
        # define the face width for the whole sequence
        facewidth = bbox.size[1]
        face = crop_face(frame, bbox, facewidth)
        
        if not wholeface:
          good_features = get_good_features_to_track(face, npoints, quality, distance, plot)
      else:
        # subsequent frames:
        # -> crop the face with the bounding_boxes of the previous frame (so
        #    that faces are of the same size)
        # -> get the projection of the corners detected in the previous frame
        # -> find the (affine) transformation relating previous corners with
        #    current corners
        # -> apply this transformation to the mask
        face = crop_face(frame, prev_bb, facewidth)
        if not wholeface:
          good_features = track_features(prev_face, face, prev_features, plot)
          project = find_transformation(prev_features, good_features)
          if project is None: 
            logger.warn("Sequence {0}, frame {1} : No projection was found"
                " between previous and current frame, mask from previous frame will be used"
                .format(obj.path, i))
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
        prev_bb = bb

      
      if not wholeface:
        prev_face = crop_face(frame, prev_bb, facewidth)
        prev_features = get_good_features_to_track(face, npoints, quality, distance, plot)
        if prev_features is None:
          logger.warn("Sequence {0}, frame {1} No features to track"  
              " detected in the current frame, using the previous ones"
              .format(obj.path, i))
          prev_features = good_features

        # get the bottom face region average colors
        face_mask = get_mask(frame, mask_points)
        # original algorithm: green only
        face_color[i] = compute_average_colors_mask(frame, face_mask, plot)[1]
      else:
        face_color[i] = compute_average_colors_wholeface(face, plot)

      # get the background region average colors
      bg_mask = numpy.zeros((frame.shape[1], frame.shape[2]), dtype=bool)
      bg_mask[:100, :100] = True
      bg_color[i] = compute_average_colors_mask(frame, bg_mask, plot)[1]

    # saves the data into an HDF5 file with a '.hdf5' extension
    out_facedir = os.path.dirname(output_face)
    if not os.path.exists(out_facedir): bob.io.base.create_directories_safe(out_facedir)
    bob.io.base.save(face_color, output_face)
    logger.info("Output file saved to `%s'...", output_face)

    out_bgdir = os.path.dirname(output_bg)
    if not os.path.exists(out_bgdir): bob.io.base.create_directories_safe(out_bgdir)
    bob.io.base.save(bg_color, output_bg)
    logger.info("Output file saved to `%s'...", output_bg)
