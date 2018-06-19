#!/usr/bin/env python
# encoding: utf-8

"""Non Rigid Motion Elimination for color signals (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot] [--indir=<path>] [--outdir=<path>]
           [--seglength=<int>] [--save-threshold=<path>] [--load-threshold=<path>]
           [--cutoff=<float>] [--cvpr14] [--overwrite]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                    Show this help message and exit
  -v, --verbose                 Increases the verbosity (may appear multiple times)
  -V, --version                 Show version
  -P, --plot                    Set this flag if you'd like to follow-up the algorithm
                                execution graphically. We'll plot some interactions.
  -p, --protocol=<string>       Protocol [default: all].
  -s, --subset=<string>         Data subset to load. If nothing is provided 
                                all the data sets will be loaded.
  -i, --indir=<path>            The path to the saved illumination corrected signal
                                on your disk [default: illumination].
  -o, --outdir=<path>           The path to the output directory where the resulting
                                motion corrected signals will be stored
                                [default: motion].
  -L, --seglength=<int>         The length of the segments [default: 61]
      --cutoff=<float>          Specify the percentage of largest segments to
                                determine the threshold [default: 0.05].
      --save-threshold=<path>   Save the found threshold to cut segments [default: threshold.txt]. 
      --load-threshold=<path>   Load the threshold to cut segments [default: None]. 
  -O, --overwrite               By default, we don't overwrite existing files. The
                                processing will skip those so as to go faster. If you
                                still would like me to overwrite them, set this flag.
  --cvpr14                      Original algorithm, as provided by the authors
                                of the paper (contains a bug, provided for
                                reproducibilty purposes). This has to be set
                                to reproduce results in the paper.


Examples:

  To run the motion elimination for the cohface database

    $ %(prog)s cohface -v

  To just run a preliminary benchmark tests on the first 10 videos, do:

    $ %(prog)s cohface -v -l 10

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


See '%(prog)s --help' for more information.

"""

import os
import sys
import pkg_resources

import bob.core
logger = bob.core.log.setup("bob.rppg.base")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

from ..motion_utils import build_segments
from ..motion_utils import prune_segments 
from ..motion_utils import build_final_signal 
from ..motion_utils import build_final_signal_cvpr14

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Non-rigid motion elimination for videos (%s)' % version,)

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

  # determine the threshold for the standard deviation to be applied to the segments
  # this part is not executed if a threshold is provided
  if (args['--load-threshold']) == 'None':
    all_stds = []
    for obj in objects:

      # load the llumination corrected signal
      logger.debug("Computing standard deviations in color signals from `%s'...", obj.path)
      illum_file = obj.make_path(args['--indir'], '.hdf5')
      try:
        color = bob.io.base.load(illum_file)
      except (IOError, RuntimeError) as e:
        logger.warn("Skipping file `%s' (no color signals file available)",  obj.path)
        continue

      # skip this file if there are NaN ...
      if numpy.isnan(numpy.sum(color)):
        logger.warn("Skipping file `%s' (NaN in file)",  obj.path)
        continue
      
      # get the standard deviation in the segments
      green_segments, __ = build_segments(color, int(args['--seglength']))
      std_green = numpy.std(green_segments, 1, ddof=1)
      all_stds.extend(std_green.tolist())

    logger.info("Standard deviations are computed")

    # sort the std and find the 5% at the top to get the threshold
    sorted_stds = sorted(all_stds, reverse=True)
    cut_index = int(float(args['--cutoff']) * len(all_stds)) + 1
    threshold = sorted_stds[cut_index]
    logger.info("The threshold was {0} (removing {1} percent of the largest segments)".format(threshold, 100*float(args['--cutoff'])))

    # write threshold to file
    f = open(args['--save-threshold'], 'w')
    f.write(str(threshold))
    f.close()

  else:
    # load threshold
    f = open(args['--load-threshold'], 'r')
    threshold = float(f.readline().rstrip())

    # cut segments where the std is too large
    for obj in objects:

      # expected output file
      output = obj.make_path(args['--outdir'], '.hdf5')

      # if output exists and not overwriting, skip this file
      if os.path.exists(output) and not args['--overwrite']:
        logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
        continue

      # load the color signals
      logger.debug("Eliminating motion in color signals from `%s'...", obj.path)
      illum_file = obj.make_path(args['--indir'], '.hdf5')
      try:
        color = bob.io.base.load(illum_file)
      except (IOError, RuntimeError) as e:
        logger.warn("Skipping file `%s' (no color signals file available)",
            obj.path)
        continue
        
      # skip this file if there are NaN ...
      if numpy.isnan(numpy.sum(color)):
        logger.warn("Skipping file `%s' (NaN in file)",  obj.path)
        continue

      # divide the signals into segments
      green_segments, end_index = build_segments(color, int(args['--seglength']))
      # remove segments with high variability
      pruned_segments, gaps, cut_index = prune_segments(green_segments, threshold)
      
      # build final signal - but be sure that there are some segments left !
      if pruned_segments.shape[0] == 0:
        logger.warn("All segments have been discared in {0}".format(obj.path))
        continue
      if bool(args['--cvpr14']):
        corrected_green = build_final_signal_cvpr14(pruned_segments, gaps)
      else:
        corrected_green = build_final_signal(pruned_segments, gaps)
     
      if bool(args['--plot']):
        from matplotlib import pyplot
        f, axarr = pyplot.subplots(2, sharex=True)
        axarr[0].plot(range(end_index), color[:end_index], 'g')
        xmax, xmin, ymax, ymin = axarr[0].axis()
        for cuts in cut_index:
          axarr[0].vlines(cuts[0], ymin, ymax, color='black', linewidths='2')
          axarr[0].vlines(cuts[1], ymin, ymax, color='black', linewidths='2')
          axarr[0].plot(range(cuts[0],cuts[1]), color[cuts[0]:cuts[1]], 'r')
        axarr[0].set_title('Original color pulse')
        axarr[1].plot(range(corrected_green.shape[0]), corrected_green, 'g')
        axarr[1].set_title('Motion corrected color pulse')
        pyplot.show()

      # saves the data into an HDF5 file with a '.hdf5' extension
      outdir = os.path.dirname(output)
      if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
      bob.io.base.save(corrected_green, output)
      logger.info("Output file saved to `%s'...", output)

  return 0
