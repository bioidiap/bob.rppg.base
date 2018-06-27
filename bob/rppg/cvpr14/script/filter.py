#!/usr/bin/env python
# encoding: utf-8

"""Filtering (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]  
           [--verbose ...] [--plot] [--motiondir=<path>] [--pulsedir=<path>]
           [--Lambda=<int>] [--window=<int>] [--framerate=<int>] [--order=<int>]
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
  -i, --motiondir=<path>        The path to the saved signals to be filtered on
                            your disk [default: motion].
  -o, --pulsedir=<path>       The path to the output directory where the resulting
                            color signals will be stored [default: filtered].
  --Lambda=<int>            Lambda parameter for detrending (see article) [default: 300]
  --window=<int>            Moving window length [default: 23]
  -f, --framerate=<int>     Frame-rate of the video sequence [default: 61]
  --order=<int>             Bandpass filter order [default: 128]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --gridcount               Tells the number of objects and exits.

Examples:

  To run the filters 

    $ %(prog)s config.py -v


See '%(prog)s --help' for more information.


Reference:

Part of this code is based on the following article 
"An advanced detrending method with application to HRV analysis". 
Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

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

from ..filter_utils import detrend
from ..filter_utils import average 
from ...base.utils import build_bandpass_filter 
from ...base.utils import get_parameter

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Filtering for signals (%s)' % version,)

  # load configuration file
  configuration = load([os.path.join(args['<configuration>'])])
  
  # get various parameters, either from config file or command-line 
  protocol = get_parameter(args, configuration, 'protocol', 'all')
  subset = get_parameter(args, configuration, 'subset', None)
  motiondir = get_parameter(args, configuration, 'motiondir', 'motion')
  pulsedir = get_parameter(args, configuration, 'pulsedir', 'pulse')
  Lambda = get_parameter(args, configuration, 'Lambda', 300)
  window = get_parameter(args, configuration, 'window', 23)
  framerate = get_parameter(args, configuration, 'framerate', 61)
  order = get_parameter(args, configuration, 'order', 128)
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

  # build the bandpass filter one and for all
  b = build_bandpass_filter(framerate, order, plot)

  ################
  ### LET'S GO ###
  ################
  for obj in objects:

    # expected output file
    output = obj.make_path(pulsedir, '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the corrected color signals of shape (3, nb_frames)
    logger.info("Filtering in signal from `%s'...", obj.path)
    motion_file = obj.make_path(motiondir, '.hdf5')
    try:
      motion_corrected_signal = bob.io.base.load(motion_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no motion corrected signal file available)", obj.path)
      continue

    # check whether the signal is long enough to be filtered with the bandpass of this order
    padlen = 3 * len(b)
    if motion_corrected_signal.shape[0] < padlen:
      logger.warn("Skipping file {0} (unable to bandpass filter it, the signal is probably not long enough)".format(obj.path))
      continue

    # detrend
    green_detrend = detrend(motion_corrected_signal, Lambda)
    # average
    green_averaged = average(green_detrend, window)
    # bandpass
    from scipy.signal import filtfilt
    green_bandpassed = filtfilt(b, numpy.array([1]), green_averaged)

    # plot the result
    if plot:
      from matplotlib import pyplot
      f, ax = pyplot.subplots(4, sharex=True)
      ax[0].plot(range(motion_corrected_signal.shape[0]), motion_corrected_signal, 'g')
      ax[0].set_title('Original signal')
      ax[1].plot(range(motion_corrected_signal.shape[0]), green_detrend, 'g')
      ax[1].set_title('After detrending')
      ax[2].plot(range(motion_corrected_signal.shape[0]), green_averaged, 'g')
      ax[2].set_title('After averaging')
      ax[3].plot(range(motion_corrected_signal.shape[0]), green_bandpassed, 'g')
      ax[3].set_title('Bandpassed signal')
      pyplot.show()

    output_data = numpy.copy(green_bandpassed)

    # saves the data into an HDF5 file with a '.hdf5' extension
    pulse_outdir = os.path.dirname(output)
    if not os.path.exists(pulse_outdir): bob.io.base.create_directories_safe(pulse_outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)
  
  return 0
