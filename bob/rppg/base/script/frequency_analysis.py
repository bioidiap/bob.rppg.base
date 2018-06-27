#!/usr/bin/env python
# encoding: utf-8

"""Frequency analysis of the pulse signal to get the heart-rate (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]  
           [--verbose ...] [--plot] [--pulsedir=<path>] [--hrdir=<path>] 
           [--framerate=<int>] [--nsegments=<int>] [--nfft=<int>] 
           [--overwrite] 

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -V, --version             Show version
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -p, --protocol=<string>   Protocol [default: all].[default: all]
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -i, --pulsedir=<path>     The path to the saved filtered signals on your disk
                            [default: pulse].
  -o, --hrdir=<path>        The path to the output directory where the resulting
                            color signals will be stored [default: hr].
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -f, --framerate=<int>     Frame-rate of the video sequence [default: 61]
  --nsegments=<int>         Number of overlapping segments in Welch procedure
                            [default: 12].
  --nfft=<int>              Number of points to compute the FFT [default: 8192].

Examples:

  To run the frequency analysis 

    $ %(prog)s config.py -v


See '%(prog)s --help' for more information.

"""

import os
import sys
import pkg_resources

from bob.core.log import setup
logger = setup("bob.rppg.base")

from docopt import docopt

from bob.extension.config import load
from ..utils import get_parameter

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

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
  pulsedir = get_parameter(args, configuration, 'pulsedir', 'pulse')
  hrdir = get_parameter(args, configuration, 'hrdir', 'hr')
  framerate = get_parameter(args, configuration, 'framerate', 61)
  nsegments = get_parameter(args, configuration, 'nsegments', 12)
  nfft = get_parameter(args, configuration, 'nfft', 8192)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  verbosity_level = get_parameter(args, configuration, 'verbose', 0)

  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, args['--verbose'])
 
  # TODO: find a way to check protocol names - Guillaume HEUSCH, 22-06-2018
  if hasattr(configuration, 'database'):
    objects = configuration.database.objects(protocol, subset)
  else:
    logger.error("Please provide a database in your configuration file !")
    sys.exit()

  ################
  ### LET'S GO ###
  ################
  for obj in objects:

    # expected output file
    output = obj.make_path(hrdir, '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the filtered color signals of shape (3, nb_frames)
    logger.info("Frequency analysis of color signals from `%s'...", obj.path)
    filtered_file = obj.make_path(pulsedir, '.hdf5')
    try:
      signal = bob.io.base.load(filtered_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no color signals file available)", obj.path)
      continue

    if plot:
      from matplotlib import pyplot
      pyplot.plot(range(signal.shape[0]), signal, 'g')
      pyplot.title('Filtered green signal')
      pyplot.show()

    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (nsegments + 1) 

    # the number of points for FFT should be larger than the segment length ...
    if nfft < segment_length:
      logger.warn("Skipping file `%s' (nfft < nperseg)", obj.path)
      continue

    from scipy.signal import welch
    green_f, green_psd = welch(signal, framerate, nperseg=segment_length, nfft=nfft)

    # find the max of the frequency spectrum in the range of interest
    first = numpy.where(green_f > 0.7)[0]
    last = numpy.where(green_f < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = numpy.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]
    hr = f_max*60.0
    logger.info("Heart rate = {0}".format(hr))

    if plot:
      from matplotlib import pyplot
      pyplot.semilogy(green_f, green_psd, 'g')
      xmax, xmin, ymax, ymin = pyplot.axis()
      pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
      pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
      pyplot.show()

    output_data = numpy.array([hr], dtype='float64')

    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)

  return 0
