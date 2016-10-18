#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 22 Oct 10:19:56 CEST 2015

"""Bounding-box extractor for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...]
           [--dbdir=<path>] [--outdir=<path>] 
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
  -o, --outdir=<path>       The path to the output directory where the resulting
                            bounding boxes will be stored [default: bboxes]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times).
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -g, --gridcount           Prints the number of objects to process and exits.



Examples:

  To run the bounding-box extractor for the cohface database

    $ %(prog)s cohface -v

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


See '%(prog)s --help' for more information.

"""

import os
import sys
import pkg_resources

import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("bbox_log")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import bob.io.base
import bob.ip.facedetect


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
      version='Bounding-box extractor for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger("bbox_log").setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger("bbox_log").setLevel(logging.DEBUG)

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

  if args['--gridcount']:
    print len(objects)
    sys.exit()

  # if we are on a grid environment, just find what I have to process.
  if os.environ.has_key('SGE_TASK_ID'):
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(objects):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(objects))
    objects = [objects[pos]]

  # does the actual work - for every video in the available dataset, calculates
  # the bounding boxes and dumps the result to the output directory
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--outdir'], '.face')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    video = obj.load_video(dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # the result
    detections = {}
    
    for i, frame in enumerate(video):
      logger.debug("Processing frame %d/%d...", i+1, len(video))

      detected = bob.ip.facedetect.detect_single_face(frame)
      if detected is not None:
        bb, quality = detected
        detections[i] = bb

        if bool(args['--plot']):
          import bob.ip.draw
          from matplotlib import pyplot
          import numpy
          bob.ip.draw.box(frame, bb.topleft, bb.size, color=(255,0,0))
          pyplot.imshow(numpy.rollaxis(numpy.rollaxis(frame, 2),2))
          pyplot.show()
      else:
        logger.debug("Could not find a face at frame %d", k)

    # saves the data into a text file with a '.face' extension, order is:
    # frame top-left-x top-left-y width height. Like for antispoofing.utils
    # all values are integers
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    with open(output, 'wt') as f:
      for k in sorted(detections.keys()):
        (y, x) = detections[k].topleft_f
        (height, width) = detections[k].size_f
        f.write('%d %f %f %f %f\n' % (k, x, y, width, height))
    logger.info("Output file saved to `%s'...", output)
