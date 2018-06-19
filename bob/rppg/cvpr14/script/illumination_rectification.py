#!/usr/bin/env python
# encoding: utf-8

"""Illumination rectification for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot]
           [--facedir=<path>][--bgdir=<path>] [--outdir=<path>] 
           [--start=<int>] [--end=<int>] [--step=<float>] 
           [--length=<int>] [--overwrite] [--gridcount]

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
  -o, --outdir=<path>       The path to the output directory where the resulting
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

  To run the illumination rectification on the hci database

    $ %(prog)s hci -v

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


See '%(prog)s --help' for more information.

"""

from __future__ import print_function

import os
import sys
import pkg_resources

import bob.core
logger = bob.core.log.setup("bob.rppg.base")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

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

  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, args['--verbose'])

  # chooses the database driver to use
  if args['cohface']:
    import bob.db.cohface
    if os.path.isdir(bob.db.cohface.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.cohface.DATABASE_LOCATION
    elif args['--facedir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--facedir']
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
    elif args['--facedir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--facedir'] 
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
    sys.exit()

  # does the actual work - for every video in the available dataset,
  # extract the average color in both the mask area and in the backround,
  # and then correct face illumination by removing the global illumination
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--outdir'], '.hdf5')
    logger.debug("expected output file -> {0}".format(output))

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the color signal of the face
    face_file = obj.make_path(args['--facedir'], '.hdf5')
    try:
      face = bob.io.base.load(face_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no face file available)", obj.path)
      continue

    # load the color signal of the background
    bg_file = obj.make_path(args['--bgdir'], '.hdf5')
    try:
      bg = bob.io.base.load(bg_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no background file available)", obj.path)
      continue

    # indices where to start and to end the processing
    logger.debug("Sequence length = {0}".format(face.shape[0]))
    start_index = int(args['--start'])
    end_index = int(args['--end'])
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
    corrected_green = rectify_illumination(face, bg, float(args['--step']), int(args['--length']))

    if bool(args['--plot']):
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
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(corrected_green, output)
    logger.info("Output file saved to `%s'...", output)

  return 0
