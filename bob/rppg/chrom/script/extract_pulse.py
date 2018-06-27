#!/usr/bin/env python
# encoding: utf-8

"""Pulse extraction using CHROM algorithm (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]
           [--pulsedir=<path>]
           [--start=<int>] [--end=<int>] [--motion=<float>]
           [--threshold=<float>] [--skininit]
           [--framerate=<int>] [--order=<int>]
           [--window=<int>] [--gridcount]
           [--overwrite] [--verbose ...] [--plot]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -o, --pulsedir=<path>     The path to the directory where signal extracted 
                            from the face area will be stored [default: pulse]
  --start=<int>             Starting frame index [default: 0].
  --end=<int>               End frame index [default: 0].
  --motion=<float>          The percentage of frames you want to select where the 
                            signal is "stable". 0 mean all the sequence [default: 0.0]. 
  --threshold=<float>       Threshold on the skin color probability [default: 0.5].
  --skininit                If you want to reinit the skin color distribution
                            at each frame.
  --framerate=<int>         Framerate of the video sequence [default: 61]
  --order=<int>             Order of the bandpass filter [default: 128]
  --window=<int>            Window size in the overlap-add procedure. A window
                            of zero means no procedure applied [default: 0].
  --gridcount               Tells the number of objects that will be processed.
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.


Example:

  To run the pulse extraction 

    $ %(prog)s config.py -v

See '%(prog)s --help' for more information.

"""
from __future__ import print_function

import os
import sys
import pkg_resources

from bob.core.log import setup
logger = setup("bob.rppg.base")

from docopt import docopt

from bob.extension.config import load
from ...base.utils import get_parameter

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base
import bob.ip.facedetect
import bob.ip.skincolorfilter

from ...base.utils import crop_face
from ...base.utils import build_bandpass_filter 

from ..extract_utils import compute_mean_rgb
from ..extract_utils import project_chrominance
from ..extract_utils import compute_gray_diff
from ..extract_utils import select_stable_frames 

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
  start = get_parameter(args, configuration, 'start', 0)
  end = get_parameter(args, configuration, 'end', 0)
  motion = get_parameter(args, configuration, 'motion', 0.0)
  threshold = get_parameter(args, configuration, 'threshold', 0.5)
  skininit = get_parameter(args, configuration, 'skininit', False)
  framerate = get_parameter(args, configuration, 'framerate', 61)
  order = get_parameter(args, configuration, 'order', 128)
  window = get_parameter(args, configuration, 'window', 0)
  overwrite = get_parameter(args, configuration, 'overwrite', False)
  plot = get_parameter(args, configuration, 'plot', False)
  gridcount = get_parameter(args, configuration, 'gridcount', False)
  verbosity_level = get_parameter(args, configuration, 'verbose', 0)
 
  # if the user wants more verbosity, lowers the logging level
  from bob.core.log import set_verbosity_level
  set_verbosity_level(logger, verbosity_level)

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
  bandpass_filter = build_bandpass_filter(framerate, order, plot)

  # does the actual work - for every video in the available dataset, 
  # extract the signals and dumps the results to the corresponding directory
  for obj in objects:

    # expected output file
    output = obj.make_path(pulsedir, '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue
    
    # load video
    video = obj.load_video(configuration.dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # indices where to start and to end the processing
    logger.debug("Sequence length = {0}".format(len(video)))
    start_index = start
    end_index = end
    if (end_index == 0):
      end_index = len(video) 
    if end_index > len(video):
      logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, len(video)))
      continue

    # number of final frames
    nb_frames = len(video)
    if end_index > 0:
      nb_frames = end_index - start_index

    # the grayscale difference between two consecutive frames (for stable frame selection)
    if motion:
      diff_motion = numpy.zeros((nb_frames-1, 1),  dtype='float64')

    # load the result of face detection
    bounding_boxes = obj.load_face_detection()

    # skin color filter
    skin_filter = bob.ip.skincolorfilter.SkinColorFilter()

    # output data
    output_data = numpy.zeros(nb_frames, dtype='float64')
    chrom = numpy.zeros((nb_frames, 2), dtype='float64')

    # loop on video frames
    counter = 0
    for i, frame in enumerate(video):

      if i >= start_index and i < end_index:
        logger.debug("Processing frame %d/%d...", i+1, end_index)

        try: 
          bbox = bounding_boxes[i]
        except NameError:
          bbox, quality = bob.ip.facedetect.detect_single_face(frame)

        # motion difference (if asked for)
        if motion > 0 and (i < (len(video) - 1)) and (counter > 0):
          current = crop_face(frame, bbox, bbox.size[1])
          diff_motion[counter-1] = compute_gray_diff(face, current)
        
        face = crop_face(frame, bbox, bbox.size[1])

        if plot and verbosity_level >= 2:
          from matplotlib import pyplot
          pyplot.imshow(numpy.rollaxis(numpy.rollaxis(face, 2),2))
          pyplot.show()

        # skin filter
        if counter == 0 or skininit:
          skin_filter.estimate_gaussian_parameters(face)
          logger.debug("Skin color parameters:\nmean\n{0}\ncovariance\n{1}".format(skin_filter.mean, skin_filter.covariance))
        skin_mask = skin_filter.get_skin_mask(face, threshold)

        if plot and verbosity_level >= 2:
          from matplotlib import pyplot
          skin_mask_image = numpy.copy(face)
          skin_mask_image[:, skin_mask] = 255
          pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
          pyplot.show()

        # sometimes skin is not detected !
        if numpy.count_nonzero(skin_mask) != 0:

          # compute the mean rgb values of the skin pixels
          r,g,b = compute_mean_rgb(face, skin_mask)
          logger.debug("Mean color -> R = {0}, G = {1}, B = {2}".format(r,g,b))

          # project onto the chrominance colorspace
          chrom[counter] = project_chrominance(r, g, b)
          logger.debug("Chrominance -> X = {0}, Y = {1}".format(chrom[counter][0], chrom[counter][1]))

        else:
          logger.warn("No skin pixels detected in frame {0}, using previous value".format(i))
          # very unlikely, but it could happened and messed up all experiments (averaging of scores ...)
          if counter == 0:
            chrom[counter] = project_chrominance(128., 128., 128.)
          else:
            chrom[counter] = chrom[counter-1]

        counter +=1
    
      elif i > end_index :
        break

    # select the most stable number of consecutive frames, if asked for
    if motion > 0:
      n_stable_frames_to_keep = int(motion * nb_frames)
      logger.info("Number of stable frames kept for motion -> {0}".format(n_stable_frames_to_keep))
      index = select_stable_frames(diff_motion, n_stable_frames_to_keep)
      logger.info("Stable segment -> {0} - {1}".format(index, index + n_stable_frames_to_keep))
      chrom = chrom[index:(index + n_stable_frames_to_keep),:]

    if plot:
      from matplotlib import pyplot
      f, axarr = pyplot.subplots(2, sharex=True)
      axarr[0].plot(range(chrom.shape[0]), chrom[:, 0], 'k')
      axarr[0].set_title("X value in the chrominance subspace")
      axarr[1].plot(range(chrom.shape[0]), chrom[:, 1], 'k')
      axarr[1].set_title("Y value in the chrominance subspace")
      pyplot.show()

    # now that we have the chrominance signals, apply bandpass
    from scipy.signal import filtfilt
    x_bandpassed = numpy.zeros(nb_frames, dtype='float64')
    y_bandpassed = numpy.zeros(nb_frames, dtype='float64')
    x_bandpassed = filtfilt(bandpass_filter, numpy.array([1]), chrom[:, 0])
    y_bandpassed = filtfilt(bandpass_filter, numpy.array([1]), chrom[:, 1])

    if plot:
      from matplotlib import pyplot
      f, axarr = pyplot.subplots(2, sharex=True)
      axarr[0].plot(range(x_bandpassed.shape[0]), x_bandpassed, 'k')
      axarr[0].set_title("X bandpassed")
      axarr[1].plot(range(y_bandpassed.shape[0]), y_bandpassed, 'k')
      axarr[1].set_title("Y bandpassed")
      pyplot.show()

    # build the final pulse signal
    alpha = numpy.std(x_bandpassed) / numpy.std(y_bandpassed)
    pulse = x_bandpassed - alpha * y_bandpassed

    # overlap-add if window_size != 0
    if window > 0:
      window_size = window
      window_stride = window_size / 2
      for w in range(0, (len(pulse)-window_size), window_stride):
        pulse[w:w+window_size] = 0.0
        xw = x_bandpassed[w:w+window_size]
        yw = y_bandpassed[w:w+window_size]
        alpha = numpy.std(xw) / numpy.std(yw)
        sw = xw - alpha * yw
        sw *= numpy.hanning(window_size)
        pulse[w:w+window_size] += sw
    
    if plot:
      from matplotlib import pyplot
      f, axarr = pyplot.subplots(1)
      pyplot.plot(range(pulse.shape[0]), pulse, 'k')
      pyplot.title("Pulse signal")
      pyplot.show()

    output_data = pulse

    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)
