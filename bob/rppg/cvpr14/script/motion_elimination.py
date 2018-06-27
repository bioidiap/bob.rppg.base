#!/usr/bin/env python
# encoding: utf-8

"""Non Rigid Motion Elimination (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot] [--illumdir=<path>] [--motiondir=<path>]
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
  -i, --illumdir=<path>            The path to the saved illumination corrected signal
                                on your disk [default: illumination].
  -o, --motiondir=<path>           The path to the output directory where the resulting
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

  To run the motion elimination 

    $ %(prog)s config.py -v


See '%(prog)s --help' for more information.

"""

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

  # load configuration file
  configuration = load([os.path.join(args['<configuration>'])])

  # get various parameters, either from config file or command-line 
  protocol = get_parameter(args, configuration, 'protocol', 'all')
  subset = get_parameter(args, configuration, 'subset', None)
  illumdir = get_parameter(args, configuration, 'illumdir', 'illumination')
  motiondir = get_parameter(args, configuration, 'motiondir', 'motion')
  seglength = get_parameter(args, configuration, 'seglength', 61)
  cutoff = get_parameter(args, configuration, 'cutoff', 0.05)
  save_threshold = get_parameter(args, configuration, 'save-threshold', 'threshold.txt')
  load_threshold = get_parameter(args, configuration, 'load-threshold', '')
  cvpr14 = get_parameter(args, configuration, 'cvpr14', False)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  verbosity_level = get_parameter(args, configuration, 'verbose', 0)
 
  print(load_threshold)
  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, args['--verbose'])

  # TODO: find a way to check protocol names - Guillaume HEUSCH, 22-06-2018
  if hasattr(configuration, 'database'):
    objects = configuration.database.objects(protocol, subset)
  else:
    logger.error("Please provide a database in your configuration file !")
    sys.exit()

  # determine the threshold for the standard deviation to be applied to the segments
  # this part is not executed if a threshold is provided
  if load_threshold == 'None':
    all_stds = []
    for obj in objects:

      # load the llumination corrected signal
      logger.debug("Computing standard deviations in color signals from `%s'...", obj.path)
      illum_file = obj.make_path(illumdir, '.hdf5')
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
      green_segments, __ = build_segments(color, seglength)
      std_green = numpy.std(green_segments, 1, ddof=1)
      all_stds.extend(std_green.tolist())

    logger.info("Standard deviations are computed")

    # sort the std and find the 5% at the top to get the threshold
    sorted_stds = sorted(all_stds, reverse=True)
    cut_index = int(cutoff * len(all_stds)) + 1
    threshold = sorted_stds[cut_index]
    logger.info("The threshold was {0} (removing {1} percent of the largest segments)".format(threshold, 100*cutoff))

    # write threshold to file
    f = open(save_threshold, 'w')
    f.write(str(threshold))
    f.close()

  else:
    # load threshold
    f = open(load_threshold, 'r')
    threshold = float(f.readline().rstrip())

    # cut segments where the std is too large
    for obj in objects:

      # expected output file
      output = obj.make_path(motiondir, '.hdf5')

      # if output exists and not overwriting, skip this file
      if os.path.exists(output) and not overwrite:
        logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
        continue

      # load the color signals
      logger.debug("Eliminating motion in color signals from `%s'...", obj.path)
      illum_file = obj.make_path(illumdir, '.hdf5')
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
      green_segments, end_index = build_segments(color, seglength)
      # remove segments with high variability
      pruned_segments, gaps, cut_index = prune_segments(green_segments, threshold)
      
      # build final signal - but be sure that there are some segments left !
      if pruned_segments.shape[0] == 0:
        logger.warn("All segments have been discared in {0}".format(obj.path))
        continue
      if cvpr14:
        corrected_green = build_final_signal_cvpr14(pruned_segments, gaps)
      else:
        corrected_green = build_final_signal(pruned_segments, gaps)
     
      if plot:
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
      outputdir = os.path.dirname(output)
      if not os.path.exists(outputdir): bob.io.base.create_directories_safe(outputdir)
      bob.io.base.save(corrected_green, output)
      logger.info("Output file saved to `%s'...", output)

  return 0
