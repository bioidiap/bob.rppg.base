#!/usr/bin/env python
# encoding: utf-8

"""Pulse extraction using CHROM algorithm (%(version)s)

Usage:
  %(prog)s <configuration>
           [--protocol=<string>] [--subset=<string> ...]
           [--pulsedir=<path>] 
           [--npoints=<int>] [--indent=<int>] [--quality=<float>] [--distance=<int>]
           [--framerate=<int>] [--order=<int>] [--window=<int>] 
           [--overwrite] [--verbose ...] [--plot] [--gridcount]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -o, --pulsedir=<path>     The path to the directory where signal extracted 
                            from the face area will be stored [default: pulse].
  -n, --npoints=<int>       Number of good features to track [default: 40]
  -i, --indent=<int>        Indent (in percent of the face width) to apply to 
                            keypoints to get the mask [default: 10]
  -q, --quality=<float>     Quality level of the good features to track
                            [default: 0.01]
  -e, --distance=<int>      Minimum distance between detected good features to
                            track [default: 10]
  --framerate=<int>         Framerate of the video sequence [default: 61]
  --order=<int>             Order of the bandpass filter [default: 128]
  --window=<int>            Window size in the overlap-add procedure. A window
                            of zero means no procedure applied [default: 0].
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -g, --gridcount           Prints the number of objects to process and exits.


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

from ...base.utils import crop_face
from ...base.utils import build_bandpass_filter

from ...cvpr14.extract_utils import kp66_to_mask
from ...cvpr14.extract_utils import get_good_features_to_track
from ...cvpr14.extract_utils import track_features
from ...cvpr14.extract_utils import find_transformation
from ...cvpr14.extract_utils import get_current_mask_points
from ...cvpr14.extract_utils import get_mask 
from ...cvpr14.extract_utils import compute_average_colors_mask

from ..extract_utils import compute_mean_rgb
from ..extract_utils import project_chrominance


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
  npoints = get_parameter(args, configuration, 'npoints', 40)
  indent = get_parameter(args, configuration, 'indent', 10)
  quality = get_parameter(args, configuration, 'quality', 0.01)
  distance = get_parameter(args, configuration, 'distance', 10)
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
    if (os.path.exists(output)) and not overwrite:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue
    
    # load video
    video = obj.load_video(configuration.dbdir)
    logger.info("Processing input video from `%s'...", video.filename)

    # number of frames
    nb_frames = len(video)
    
    # load the result of face detection
    bounding_boxes = obj.load_face_detection()

    # output data
    output_data = numpy.zeros(nb_frames, dtype='float64')
    chrom = numpy.zeros((nb_frames, 2), dtype='float64')

    # loop on video frames
    for i, frame in enumerate(video):
      logger.debug("Processing frame %d/%d...", i+1, len(video))

      if i == 0:
        # first frame:
        # -> load the keypoints detected by DMRF
        # -> infer the mask from the keypoints
        # -> detect the face
        # -> get "good features" inside the face
        kpts = obj.load_drmf_keypoints()
        mask_points, mask = kp66_to_mask(frame, kpts, int(indent), plot)

        try: 
          bbox = bounding_boxes[i]
        except NameError:
          bbox, quality = bob.ip.facedetect.detect_single_face(frame)
        
        # define the face width for the whole sequence
        facewidth = bbox.size[1]
        face = crop_face(frame, bbox, facewidth)
        good_features = get_good_features_to_track(face,npoints, quality, distance, plot)
      else:
        # subsequent frames:
        # -> crop the face with the bounding_boxes of the previous frame (so
        #    that faces are of the same size)
        # -> get the projection of the corners detected in the previous frame
        # -> find the (affine) transformation relating previous corners with
        #    current corners
        # -> apply this transformation to the mask
        face = crop_face(frame, prev_bb, facewidth)
        good_features = track_features(prev_face, face, prev_features, plot)
        project = find_transformation(prev_features, good_features)
        if project is None: 
          logger.warn("Sequence {0}, frame {1} : No projection was found"
              " between previous and current frame, mask from previous frame will be used"
                .format(obj.path, i))
        else:
          mask_points = get_current_mask_points(mask_points, project)

      # update stuff for the next frame:
      # -> the previous face is the face in this frame, with its bbox (and not
      #    with the previous one)
      # -> the features to be tracked on the next frame are re-detected
      try: 
        prev_bb = bounding_boxes[i]
      except NameError:
        bb, quality = bob.ip.facedetect.detect_single_face(frame)
        prev_bb = bb
      
      prev_face = crop_face(frame, prev_bb, facewidth)
      prev_features = get_good_features_to_track(face, npoints, quality, distance, plot)
      if prev_features is None:
        logger.warn("Sequence {0}, frame {1} No features to track"  
            " detected in the current frame, using the previous ones"
            .format(obj.path, i))
        prev_features = good_features

      # get the mask 
      face_mask = get_mask(frame, mask_points)
      if plot and verbosity_level >= 2:
        from matplotlib import pyplot
        mask_image = numpy.copy(frame)
        mask_image[:, face_mask] = 255
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(mask_image, 2),2))
        pyplot.show()


      # compute the mean rgb values of the pixels inside the mask
      r,g,b = compute_mean_rgb(frame, face_mask)
      # project onto the chrominance colorspace
      chrom[i] = project_chrominance(r, g, b)


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
    if int(window) > 0:
      window_size = int(window)
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
    out_pulsedir = os.path.dirname(output)
    if not os.path.exists(out_pulsedir): bob.io.base.create_directories_safe(out_pulsedir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)
