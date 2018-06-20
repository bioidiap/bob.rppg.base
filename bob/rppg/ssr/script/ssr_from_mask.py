#!/usr/bin/env python
# encoding: utf-8

"""Pulse extractor using 2SR algorithm (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...]
           [--dbdir=<path>] [--pulsedir=<path>] 
           [--npoints=<int>] [--indent=<int>] [--quality=<float>] [--distance=<int>]
           [--stride=<int>] 
           [--overwrite] [--verbose ...] [--plot] [--gridcount]

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
  -o, --pulsedir=<path>     The path to the directory where signal extracted 
                            from the face area will be stored [default: face]
  -n, --npoints=<int>       Number of good features to track [default: 40]
  -i, --indent=<int>        Indent (in percent of the face width) to apply to 
                            keypoints to get the mask [default: 10]
  -q, --quality=<float>     Quality level of the good features to track
                            [default: 0.01]
  -e, --distance=<int>      Minimum distance between detected good features to
                            track [default: 10]
  --stride=<int>            Temporal stride [default: 20]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -g, --gridcount           Prints the number of objects to process and exits.


Example:

  To run the pulse extractor for the cohface database

    $ %(prog)s cohface -v

See '%(prog)s --help' for more information.

"""
from __future__ import print_function

import os
import sys
import pkg_resources

from bob.core.log import setup
logger = setup("bob.rppg.base")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base
import bob.ip.facedetect

from ...base.utils import crop_face

from ...cvpr14.extract_utils import kp66_to_mask
from ...cvpr14.extract_utils import get_good_features_to_track
from ...cvpr14.extract_utils import track_features
from ...cvpr14.extract_utils import find_transformation
from ...cvpr14.extract_utils import get_current_mask_points
from ...cvpr14.extract_utils import get_mask 
from ...cvpr14.extract_utils import compute_average_colors_mask

from ..ssr_utils import get_eigen
from ..ssr_utils import plot_eigenvectors
from ..ssr_utils import build_P 


def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Signal extractor for videos (%s)' % version,)

  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, args['--verbose'])

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

  if args['--gridcount']:
    print(len(objects))

  # the temporal stride
  temporal_stride = int(args['--stride'])

  # does the actual work - for every video in the available dataset, 
  # extract the signals and dumps the results to the corresponding directory
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--pulsedir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if (os.path.exists(output)) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue
    
    # load video
    video = obj.load_video(dbdir)
    logger.info("Processing input video from `%s'...", video.filename)
    nb_final_frames = len(video)

    # load the result of face detection
    bounding_boxes = obj.load_face_detection()

    # the result -> the pulse signal 
    output_data = numpy.zeros(nb_final_frames, dtype='float64')
    
    # store the eigenvalues and the eigenvectors at each frame 
    eigenvalues = numpy.zeros((3, nb_final_frames), dtype='float64')
    eigenvectors = numpy.zeros((3, 3, nb_final_frames), dtype='float64')

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
        
        # define the face width for the whole sequence
        facewidth = bbox.size[1]
        face = crop_face(frame, bbox, facewidth)
        
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
        face = crop_face(frame, prev_bb, facewidth)
        good_features = track_features(prev_face, face, prev_features, bool(args['--plot']))
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

      
      prev_face = crop_face(frame, prev_bb, facewidth)
      prev_features = get_good_features_to_track(face, int(args['--npoints']),
          float(args['--quality']), int(args['--distance']),
          bool(args['--plot']))
      if prev_features is None:
        logger.warn("Sequence {0}, frame {1} No features to track"  
            " detected in the current frame, using the previous ones"
            .format(obj.path, i))
        prev_features = good_features

      # get the bottom face region
      face_mask = get_mask(frame, mask_points)

      if bool(args['--plot']):
        from matplotlib import pyplot
        mask_image = numpy.copy(frame)
        mask_image[:, face_mask] = 255
        pyplot.title("mask pixels in frame {0}".format(i))
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(mask_image, 2),2))
        pyplot.show()

      # get the skin pixels inside the region
      skin_pixels = frame[:, face_mask]
      skin_pixels = skin_pixels.astype('float64') / 255.0

      # build c matrix and get eigenvectors and eigenvalues
      eigenvalues[:, i], eigenvectors[:, :, i] = get_eigen(skin_pixels)

      # plot the cluster of skin pixels and eigenvectors (see Figure 1) 
      if bool(args['--plot'])  and args['--verbose'] >= 2:
        plot_eigenvectors(skin_pixels, eigenvectors[:, :, i])

      # build P and add it to the pulse signal
      if i >= temporal_stride:
        tau = i - temporal_stride
        p = build_P(i, int(args['--stride']), eigenvectors, eigenvalues)
        output_data[tau:i] += (p - numpy.mean(p)) 
        
    # plot the pulse signal
    if bool(args['--plot']):
      import matplotlib.pyplot as plt
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.plot(range(nb_final_frames), output_data)
      plt.show()

    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)

  return 0
