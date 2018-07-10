#!/usr/bin/env python
# encoding: utf-8

"""Skin color extraction for database videos (%(version)s)

Usage:
  %(prog)s <configuration> [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot]
           [--skindir=<path>] 
           [--overwrite] [--threshold=<float>] [--skininit]
           [--gridcount] 

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
  -o, --skindir=<path>       Where the skin color will be stored [default: skin].
                            database (for testing purposes)
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --threshold=<float>       Threshold on the skin probability map [default: 0.5].
  --skininit                If you want to reinitialize the skin model at each frame.
  --gridcount               Tells the number of objects and exits.



Examples:

  To run the skin color extraction


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
import bob.ip.skincolorfilter

from bob.extension.config import load
from ...base.utils import get_parameter

from ...base.utils import crop_face
from ..extract_utils import compute_average_colors_mask

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
  skindir = get_parameter(args, configuration, 'skindir', 'skin')
  threshold = get_parameter(args, configuration, 'threshold', 0.5)
  skininit = get_parameter(args, configuration, 'skininit', False)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  gridcount = get_parameter(args, configuration, 'gridcount', False)
  verbosity_level = get_parameter(args, configuration, 'verbose', 0)
  
  print(protocol)
  print(type(protocol))
  
  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, verbosity_level)

  if hasattr(configuration, 'database'):
    objects = configuration.database.objects(protocol, subset)
  else:
    logger.error("Please provide a database in your configuration file !")
    sys.exit()

  print(objects)

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
  # extract the average color in both the mask area and in the backround,
  # and then correct face illumination by removing the global illumination
  for obj in objects:

    # expected output face file
    output = obj.make_path(skindir, '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the video sequence into a reader
    video = obj.load_video(configuration.dbdir)

    logger.info("Processing input video from `%s'...", video.filename)
    logger.debug("Sequence length = {0}".format(video.number_of_frames))

    # load the result of face detection
    bounding_boxes = obj.load_face_detection() 

    # average colors of the skin color 
    skin_filter = bob.ip.skincolorfilter.SkinColorFilter()
    skin_colors = numpy.zeros((len(video), 3), dtype='float64')

    ################
    ### LET'S GO ###
    ################
    for i, frame in enumerate(video):

      logger.debug("Processing frame %d / %d...", i, len(video))
     
      facewidth = bounding_boxes[i].size[1]
      face = crop_face(frame, bounding_boxes[i], facewidth)

      # skin filter
      if i == 0 or bool(skininit):
        skin_filter.estimate_gaussian_parameters(face)
        logger.debug("Skin color parameters:\nmean\n{0}\ncovariance\n{1}".format(skin_filter.mean, skin_filter.covariance))
      skin_mask = skin_filter.get_skin_mask(face, threshold)

      if plot and i == 0:
        from matplotlib import pyplot
        skin_mask_image = numpy.copy(face)
        skin_mask_image[:, skin_mask] = 255
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
        pyplot.show()

      if numpy.count_nonzero(skin_mask) != 0:
        # green only
        skin_colors[i] = compute_average_colors_mask(face, skin_mask)[1]
      else:
        logger.warn("No skin pixels detected in frame {0}, using previous value".format(i))
        if i == 0:
          skin_colors[i] = project_chrominance(128., 128., 128.)
        else:
          skin_colors[i] = skin_colors[i-1]

    if plot:
      from matplotlib import pyplot

      f, axarr = pyplot.subplots(3, sharex=True)
      axarr[0].plot(range(skin_colors.shape[0]), skin_colors[:, 0], 'r')
      axarr[0].set_title("Average red value of the skin pixels")
      axarr[1].plot(range(skin_colors.shape[0]), skin_colors[:, 1], 'g')
      axarr[1].set_title("Average green value of the skin pixels")
      axarr[2].plot(range(skin_colors.shape[0]), skin_colors[:, 2], 'b')
      axarr[2].set_title("Average blue value of the skin pixels")

      pyplot.show()

    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(skin_colors[:, 1], output)
    logger.info("Output file saved to `%s'...", output)

  return 0
