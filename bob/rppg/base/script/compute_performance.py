#!/usr/bin/env python
# encoding: utf-8

"""Generate results from heart rate computation (%(version)s)
  
Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot] [--resultdir=<path>] [--hrdir=<path>] 
           [--overwrite] 

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -V, --version             Show version
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to see some plots. 
  -p, --protocol=<string>   Protocol [default: all].[default: all]
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -i, --hrdir=<path>        The path to the saved heart rate values on your disk [default: hr]. 
  -o, --resultdir=<path>    The path to the output directory where the results
                            will be stored [default: results].
  -O, --overwrite           By default, we don't overwrite existing files. 
                            Set this flag if you want to overwrite existing files.

Examples:

  To run the results generation 

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
  hrdir = get_parameter(args, configuration, 'hrdir', 'hr')
  resultdir = get_parameter(args, configuration, 'resultdir', 'results')
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

  # errors
  errors = []
  rmse = 0;
  mean_error_percentage = 0

  inferred = []
  ground_truth = []

  ################
  ### LET'S GO ###
  ################
  
  # if output dir exists and not overwriting, stop 
  if os.path.exists(resultdir) and not overwrite:
    logger.info("Skipping output `%s': already exists, use --overwrite to force an overwrite", resultdir)
    sys.exit()
  else: 
    bob.io.base.create_directories_safe(resultdir)

  for obj in objects:

    # load the heart rate 
    logger.debug("Loading computed heart rate from `%s'...", obj.path)
    hr_file = obj.make_path(hrdir, '.hdf5')
    try:
      hr = bob.io.base.load(hr_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no heart rate file available)", obj.path)
      continue

    hr = hr[0]
    logger.debug("Computed heart rate : {0}".format(hr))

    # load ground truth
    gt = obj.load_heart_rate_in_bpm()
    logger.debug("Real heart rate : {0}".format(gt))
    ground_truth.append(gt)
    inferred.append(hr)
    error = hr - gt
    logger.debug("Error = {0}".format(error))
    errors.append(error)
    rmse += error**2
    mean_error_percentage += numpy.abs(error)/gt

  # compute global statistics 
  rmse /= len(errors)
  rmse = numpy.sqrt(rmse)
  rmse_text = "Root Mean Squared Error = {0:.2f}". format(rmse)
  mean_error_percentage /= len(errors)
  mean_err_percent_text = "Mean of error-rate percentage = {0:.2f}". format(mean_error_percentage)
  from scipy.stats import pearsonr
  correlation, p = pearsonr(inferred, ground_truth)
  pearson_text = "Pearson's correlation = {0:.2f} ({1:.2f} significance)". format(correlation, p)
 
  logger.info("==================")
  logger.info("=== STATISTICS ===")
  logger.info(rmse_text)
  logger.info(mean_err_percent_text)
  logger.info(pearson_text)

  # statistics in a text file
  stats_filename = os.path.join(resultdir, 'stats.txt')
  stats_file = open(stats_filename, 'w')
  stats_file.write(rmse_text + "\n")
  stats_file.write(mean_err_percent_text + "\n")
  stats_file.write(pearson_text + "\n")
  stats_file.close()

  # scatter plot
  from matplotlib import pyplot
  f = pyplot.figure()
  ax = f.add_subplot(1,1,1)
  ax.scatter(ground_truth, inferred)
  ax.plot([40, 110], [40, 110], 'r--', lw=2)
  pyplot.xlabel('Ground truth [bpm]')
  pyplot.ylabel('Estimated heart-rate [bpm]')
  ax.set_title('Scatter plot')
  scatter_file = os.path.join(resultdir, 'scatter.png')
  pyplot.savefig(scatter_file)

  # histogram of error
  f2 = pyplot.figure()
  ax2 = f2.add_subplot(1,1,1)
  ax2.hist(errors, bins=50, )
  ax2.set_title('Distribution of the error')
  distribution_file = os.path.join(resultdir, 'error_distribution.png')
  pyplot.savefig(distribution_file)

  # distribution of HR
  f3 = pyplot.figure()
  ax3 = f3.add_subplot(1,1,1)
  histoargs = {'bins': 50, 'alpha': 0.5, 'histtype': 'bar', 'range': (30, 120)} 
  pyplot.hist(ground_truth, label='Real HR', color='g', **histoargs)
  pyplot.hist(inferred, label='Estimated HR', color='b', **histoargs)
  pyplot.ylabel("Test set")

  if plot:
    pyplot.show()
  
  return 0
