#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
# 
# This file is part of bob.rpgg.base.
# 
# bob.rppg.base is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# bob.rppg.base is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bob.rppg.base. If not, see <http://www.gnu.org/licenses/>.

'''Skin color extraction for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot]
           [--dbdir=<path>] [--outdir=<path>] [--limit=<int>]
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
  -D, --dbdir=<path>        The path to the database on your disk. If not set,
                            defaults to Idiap standard locations.
  -o, --outdir=<path>       Where the skin color will be stored [default: skin].
  -l, --limit=<int>         Limits the processing to the first N videos of the
                            database (for testing purposes)
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --threshold=<float>       Threshold on the skin probability map [default: 0.5].
  --skininit                If you want to reinitialize the skin model at each frame.
  --gridcount               Tells the number of objects and exits.




Examples:

  To run the skin color extraction on the hci database

    $ %(prog)s hci -v

  To just run a preliminary benchmark tests on the first 10 videos, do:

    $ %(prog)s cohface -v -l 10

  You can change the output directory using the `-f' r `-b' flags (see help):

    $ %(prog)s hci -v -f /path/to/result/face-directory


See '%(prog)s --help' for more information.

'''

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
import bob.ip.skincolorfilter

from ...base.utils import crop_face
from ..extract_utils import compute_average_colors_mask

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
      version='Skin color extraction for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger().setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger().setLevel(logging.DEBUG)

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

  # tells the number of grid objects, and exit
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

  else: #maybe the user wants to limit the total amount of videos for testing
    if args['--limit']: objects = objects[:int(args['--limit'])]

  # does the actual work - for every video in the available dataset,
  # extract the average color in both the mask area and in the backround,
  # and then correct face illumination by removing the global illumination
  for obj in objects:

    # expected output face file
    output = obj.make_path(args['--outdir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the video sequence into a reader
    video = obj.load_video(dbdir)

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
      if i == 0 or bool(args['--skininit']):
        skin_filter.estimate_gaussian_parameters(face)
        logger.debug("Skin color parameters:\nmean\n{0}\ncovariance\n{1}".format(skin_filter.mean, skin_filter.covariance))
      skin_mask = skin_filter.get_skin_mask(face, float(args['--threshold']))

      if bool(args['--plot']) and i == 0:
        from matplotlib import pyplot
        skin_mask_image = numpy.copy(face)
        skin_mask_image[:, skin_mask] = 255
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
        pyplot.show()

      if numpy.count_nonzero(skin_mask) != 0:
        skin_colors[i] = compute_average_colors_mask(face, skin_mask)
      else:
        logger.warn("No skin pixels detected in frame {0}, using previous value".format(i))
        if i == 0:
          skin_colors[i] = project_chrominance(128., 128., 128.)
        else:
          skin_colors[i] = skin_colors[i-1]

    if bool(args['--plot']):
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
