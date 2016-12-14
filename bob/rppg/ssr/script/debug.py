#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Wed 11 May 09:31:36 CEST 2016

'''Debug for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot]
           [--mydir=<path>] [--refdir=<path>] 
          
  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -V, --version             Show version
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -D, --mydir=<path>        The path to the data I generated
  -b, --refdir=<path>       The path to the reference data.
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.


Examples:

  To run the debug process on the hci database

    $ %(prog)s hci -v

See '%(prog)s --help' for more information.

'''

import os
import sys
import pkg_resources


import logging
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("debug_logger")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.video
import bob.ip.color

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
      version='Skin pixels extraction for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger("ssr_logger").setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger("ssr_logger").setLevel(logging.DEBUG)

  # chooses the database driver to use
  if args['cohface']:
    import bob.db.cohface
    if os.path.isdir(bob.db.cohface.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.cohface.DATABASE_LOCATION
    elif args['--dbdir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--dbdir']
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
    elif args['--dbdir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--dbdir'] 
    else:
      logger.warn("Could not find the database directory, please provide one")
      sys.exit()
    db = bob.db.hci_tagging.Database()
    if not((args['--protocol'] == 'all') or (args['--protocol'] == 'cvpr14')):
      logger.warning("Protocol should be either 'all' or 'cvpr14' (and not {0})".format(args['--protocol']))
      sys.exit()
    objects = db.objects(args['--protocol'], args['--subset'])

  
  # does the actual work 
  sum_error = 0.
  n_errors = 0
  n_sequences = 0
  for obj in objects:

    # load my data 
    my_file = obj.make_path(args['--mydir'], '.hdf5')
    try:
      mydata = bob.io.base.load(my_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no my data file available)", my_file)
      continue

    # load ref data 
    ref_file = obj.make_path(args['--refdir'], '.hdf5')
    try:
      refdata = bob.io.base.load(ref_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no ref data file available)", ref_file)
      continue

    assert refdata.shape == mydata.shape, "not the same size !"
    n_sequences += 1

    if numpy.mean(numpy.abs(refdata - mydata)) > 0:
     
      error = numpy.mean(numpy.abs(refdata - mydata))
      print "error = {0}".format(error)
      sum_error += error 

      if error > 10e-10:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(mydata.shape[0]), mydata, 'r')
        ax.plot(range(refdata.shape[0]), refdata, 'k')
        plt.show()
        print my_file
        n_errors += 1
        #os.remove(my_file)
    else:
      print "No error :)"

  print "mean error = {0} ({1} / {2})".format((sum_error / n_sequences), sum_error, n_sequences)
  print "number of significant errors = {0}".format(n_errors)


  return 0
