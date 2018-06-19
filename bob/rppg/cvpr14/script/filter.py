#!/usr/bin/env python
# encoding: utf-8

"""Filtering of color signal (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...]  
           [--verbose ...] [--plot] [--indir=<path>] [--outdir=<path>]
           [--lambda=<int>] [--window=<int>] [--framerate=<int>] [--order=<int>]
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
  -i, --indir=<path>        The path to the saved signals to be filtered on
                            your disk [default: motion].
  -o, --outdir=<path>       The path to the output directory where the resulting
                            color signals will be stored [default: filtered].
  --lambda=<int>            Lambda parameter for detrending (see article) [default: 300]
  --window=<int>            Moving window length [default: 23]
  -f, --framerate=<int>     Frame-rate of the video sequence [default: 61]
  --order=<int>             Bandpass filter order [default: 128]
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --gridcount               Tells the number of objects and exits.

Examples:

  To run the filters on the cohface database

    $ %(prog)s cohface -v

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


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

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

from ..filter_utils import detrend
from ..filter_utils import average 
from ...base.utils import build_bandpass_filter 

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
  else:
      arguments = sys.argv[1:]

  prog = os.path.basename(sys.argv[0])
  completions = dict(prog=prog, version=version,)
  args = docopt(__doc__ % completions, argv=arguments, version='Filtering for signals (%s)' % version,)

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

  # build the bandpass filter one and for all
  b = build_bandpass_filter(float(args['--framerate']), int(args['--order']), bool(args['--plot']))

  ################
  ### LET'S GO ###
  ################
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--outdir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the corrected color signals of shape (3, nb_frames)
    logger.info("Filtering in signal from `%s'...", obj.path)
    motion_file = obj.make_path(args['--indir'], '.hdf5')
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
    green_detrend = detrend(motion_corrected_signal, int(args['--lambda']))
    # average
    green_averaged = average(green_detrend, int(args['--window']))
    # bandpass
    from scipy.signal import filtfilt
    green_bandpassed = filtfilt(b, numpy.array([1]), green_averaged)

    # plot the result
    if bool(args['--plot']):
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
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)
  
  return 0
