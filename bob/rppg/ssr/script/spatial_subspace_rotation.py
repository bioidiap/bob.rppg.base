#!/usr/bin/env python
# encoding: utf-8

"""Pulse extraction using 2SR algorithm (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot]
           [--dbdir=<path>] [--outdir=<path>]
           [--threshold=<float>] [--skininit] 
           [--stride=<int>] [--start=<int>] [--end=<int>] 
           [--overwrite] [--gridcount]
          
  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -V, --version             Show version
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -D, --dbdir=<path>        The path to the database on your disk. If not set,
                            defaults to Idiap standard locations.
  -o, --outdir=<path>       The path to the output directory where the resulting
                            pulse signal will be stored [default: pulse].
  --threshold=<float>       Threshold on the skin probability map [default: 0.5].
  --skininit                If you want to reinitialize the skin model at each frame.
  -s, --start=<int>         Index of the starting frame [default: 0].
  -e, --end=<int>           Index of the ending frame. If set to zero, the
                            processing will be done to the last frame [default: 0].
  --stride=<int>            Temporal stride [default: 61]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --gridcount               Tells the number of objects and exits.


Examples:

  To run the spatial subspace rotation on the hci database

    $ %(prog)s hci -v

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


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
import bob.io.video
import bob.ip.color

from ...base.utils import crop_face
from ..ssr_utils import get_skin_pixels
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
    elif args['--dbdir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--dbdir']
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
    elif args['--dbdir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--dbdir'] 
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

  # does the actual work 
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--outdir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the video sequence into a reader
    video = obj.load_video(dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # indices where to start and to end the processing
    logger.debug("Sequence length = {0}".format(len(video)))
    start_index = int(args['--start'])
    end_index = int(args['--end'])
    if (end_index == 0):
      end_index = len(video) 
    if end_index > len(video):
      logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, len(video)))
      continue
    
    # truncate the signals if needed
    nb_final_frames = end_index - start_index
    logger.debug("Processing %d frames...", nb_final_frames)

    # load the result of face detection
    bounding_boxes = obj.load_face_detection()

    # the temporal stride
    temporal_stride = int(args['--stride'])

    # the result -> the pulse signal 
    output_data = numpy.zeros(nb_final_frames, dtype='float64')

    # store the eigenvalues and the eigenvectors at each frame 
    eigenvalues = numpy.zeros((3, nb_final_frames), dtype='float64')
    eigenvectors = numpy.zeros((3, 3, nb_final_frames), dtype='float64')

    ################
    ### LET'S GO ###
    ################
    counter = 0
    for i, frame in enumerate(video):

      if i >= start_index and i < end_index:

        logger.debug("Processing frame %d/%d...", i, nb_final_frames)

        # get skin colored pixels
        try:
          if counter == 0:
            # init skin parameters in any cases if it's the first frame
            skin_pixels = get_skin_pixels(frame, i, 
               True, float(args['--threshold']), bounding_boxes)
          else:
            skin_pixels = get_skin_pixels(frame, i, 
               bool(args['--skininit']), float(args['--threshold']), bounding_boxes)
        except NameError:
          if counter == 0:
            skin_pixels = get_skin_pixels(frame, i, 
               bool(args['--skininit']), float(args['--threshold']))
          else:
            skin_pixels = get_skin_pixels(frame, i, 
               bool(args['--skininit']), float(args['--threshold']))
        logger.debug("There are {0} skin pixels in this frame".format(skin_pixels.shape[1]))
        
        # no skin pixels detected, generally due to no face detection
        # go back in time to find a face, and use this bbox to retrieve skin pixels in current frame
        if skin_pixels.shape[1] == 0:
          logger.warn("No skin pixels detected in frame {0}".format(i))
          k = 1
          while skin_pixels.shape[1] <= 0:
            
            try:
              skin_pixels = get_skin_pixels(video[i-k], (i-k),  
                 bool(args['--skininit']), float(args['--threshold']), bounding_boxes, skin_frame=frame)
            except NameError:
              skin_pixels = get_skin_pixels(video[i-k], (i-k), 
                 bool(args['--skininit']), float(args['--threshold']), skin_frame=frame)
            
            k += 1
          logger.warn("got skin pixels in frame {0}".format(i-k))

        # build c matrix and get eigenvectors and eigenvalues
        eigenvalues[:, counter], eigenvectors[:, :, counter] = get_eigen(skin_pixels)


        # plot the cluster of skin pixels and eigenvectors (see Figure 1) 
        if bool(args['--plot'])  and args['--verbose'] >= 2:
          plot_eigenvectors(skin_pixels, eigenvectors[:, :, counter])

        # build P and add it to the pulse signal
        if counter >= temporal_stride:
          tau = counter - temporal_stride
          p = build_P(counter, int(args['--stride']), eigenvectors, eigenvalues)
          output_data[tau:counter] += (p - numpy.mean(p)) 
         
        counter += 1

      elif i > end_index :
        break

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
