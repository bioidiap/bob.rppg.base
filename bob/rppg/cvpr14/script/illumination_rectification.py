#!/usr/bin/env python
# encoding: utf-8

"""Illumination rectification (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...] 
           [--facedir=<path>][--bgdir=<path>] [--illumdir=<path>] 
           [--start=<int>] [--end=<int>] [--step=<float>] 
           [--length=<int>] [--overwrite] [--gridcount]
           [--verbose ...] [--plot]

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
  -f, --facedir=<path>      The path to the directory containing the average
                            green color on the face region [default: face].
  -b, --bgdir=<path>        The path to the directory containing the average
                            green color on the background [default: background]
  -o, --illumdir=<path>       The path to the output directory where the resulting
                            corrected signal will be stored [default: illumination]
  -s, --start=<int>         Index of the starting frame [default: 0].
  -e, --end=<int>           Index of the ending frame. If set to zero, the
                            processing will be done to the last frame [default: 0].
  --step=<float>            Adaptation step of the filter weights [default: 0.05].
  --length=<int>            Length of the filter [default: 1].
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --gridcount               Tells the number of objects and exits.

Examples:

  To run the illumination rectification 

    $ %(prog)s config.py -v


See '%(prog)s --help' for more information.

"""

from __future__ import print_function

import os
import sys
import pkg_resources

import bob.core
logger = bob.core.log.setup("bob.rppg.base")

from docopt import docopt

from bob.extension.config import load

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

from ...base.utils import get_parameter
from ..illum_utils import rectify_illumination

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Illumination rectification for videos (%s)' % version,)

  # load configuration file
  configuration = load([os.path.join(args['<configuration>'])])

  # get various parameters, either from config file or command-line 
  protocol = get_parameter(args, configuration, 'protocol', 'all')
  subset = get_parameter(args, configuration, 'subset', None)
  facedir = get_parameter(args, configuration, 'facedir', 'face')
  bgdir = get_parameter(args, configuration, 'bgdir', 'bg')
  illumdir = get_parameter(args, configuration, 'illumdir', 'illumination')
  start = get_parameter(args, configuration, 'start', 0)
  end = get_parameter(args, configuration, 'end', 0)
  step = get_parameter(args, configuration, 'step', 0.05)
  length = get_parameter(args, configuration, 'length', 1)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  gridcount = get_parameter(args, configuration, 'gridcount', False)
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
  # extract the average color in both the mask area and in the backround,
  # and then correct face illumination by removing the global illumination
  for obj in objects:

    # expected output file
    output = obj.make_path(illumdir, '.hdf5')
    logger.debug("expected output file -> {0}".format(output))

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the color signal of the face
    face_file = obj.make_path(facedir, '.hdf5')
    try:
      face = bob.io.base.load(face_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no face file available)", obj.path)
      continue

    # load the color signal of the background
    bg_file = obj.make_path(bgdir, '.hdf5')
    try:
      bg = bob.io.base.load(bg_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no background file available)", obj.path)
      continue

    # indices where to start and to end the processing
    logger.debug("Sequence length = {0}".format(face.shape[0]))
    start_index = start
    end_index = end
    if (end_index == 0):
      end_index = face.shape[0]
    if end_index > face.shape[0]:
      logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, face.shape[0]))
      continue

    logger.info("Processing sequence {0} ...".format(obj.path))

    # truncate the signals if needed
    face = face[start_index:end_index]
    bg = bg[start_index:end_index]
    logger.debug("Processing %d frames...", face.shape[0])

    # apply NLMS filtering
    corrected_green = rectify_illumination(face, bg, step, length)

    if plot:
      from matplotlib import pyplot
      f, axarr = pyplot.subplots(3, sharex=True)
      axarr[0].plot(range(face.shape[0]), face, 'g')
      axarr[0].set_title(r"$g_{face}$: average green value on the mask region")
      axarr[1].plot(range(bg.shape[0]), bg, 'g')
      axarr[1].set_title(r"$g_{bg}$: average green value on the background")
      axarr[2].plot(range(corrected_green.shape[0]), corrected_green, 'g')
      axarr[2].set_title(r"$g_{IR}$: illumination rectified signal")
      pyplot.show()

    # saves the data into an HDF5 file with a '.hdf5' extension
    outputdir = os.path.dirname(output)
    if not os.path.exists(outputdir): bob.io.base.create_directories_safe(outputdir)
    bob.io.base.save(corrected_green, output)
    logger.info("Output file saved to `%s'...", output)

  return 0
