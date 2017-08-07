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

import os, sys

### WARNING ###
# be sure to run li-cohface-clean.py first to get the threshold file

# directories and file
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(current_folder)
bin_folder = os.path.join(root_folder, 'bin/')

base_expe_dir = os.path.join(root_folder, 'experiments/paper/li-xdb-cohface-hci/')
facedir = base_expe_dir + 'face'
bgdir = base_expe_dir + 'bg'
illumination_dir = base_expe_dir + 'illumination'
motion_dir = base_expe_dir + 'motion'
filtered_dir = base_expe_dir + 'filtered'
hr_dir = base_expe_dir + 'hr'
results_dir_train = base_expe_dir + 'results-train'
results_dir_test = base_expe_dir + 'results-test'

framerate = 61

# parameters
npoints = 100 
indent = 10

adaptation = 0.01
filter_length = 5

segment_length = 120 # 2 times the framerate, as for COHFACE training set
cutoff = 0.02
threshold_file = 'experiments/paper/li-cohface-clean/illumination/threshold.txt'

Lambda = 300
window = 9
order = 96

n_segments = 4
nfft = 4096

# write a file with the parameters - useful to keep track sometimes ..
param_file = base_expe_dir + '/parameters.txt'
if not os.path.isdir(base_expe_dir):
  os.makedirs(base_expe_dir)

f = open(param_file, 'w')
f.write('npoints = ' + str(npoints) + '\n')
f.write('indent [%] = ' + str(indent) + '\n\n')
f.write('adaptation step = ' + str(adaptation) + '\n')
f.write('filter length = ' + str(filter_length) + '\n\n')
f.write('segment length [frames] = ' + str(segment_length) + '\n')
f.write('cutoff [% / 100] = ' + str(cutoff) + '\n\n')
f.write('lambda = ' + str(Lambda) + '\n')
f.write('window = ' + str(window) + '\n')
f.write('order = ' + str(order) + '\n\n')
f.write('Welch segments = ' + str(n_segments) + '\n')
f.write('npoints FFT = ' + str(nfft) + '\n')
f.close()

# signals extraction
os.system(bin_folder + 'cvpr14_extract_signals.py hci --dbdir hci --subset test --bboxdir bounding-boxes --facedir ' + str(facedir) + ' --bgdir ' + str(bgdir) + ' --npoints ' + str(npoints) ' --indent ' + str(indent))

# illumination correction
os.system(bin_folder + 'cvpr14_illumination.py hci --subset test --facedir ' + facedir + ' --bgdir ' + bgdir + ' --outdir ' + illumination_dir + ' --step ' + str(adaptation) + ' --length ' + str(filter_length) + ' -v')

# motion elimination -> remove segments
os.system(bin_folder + 'cvpr14_motion.py hci --subset test --indir ' + illumination_dir + ' --outdir ' + motion_dir + ' --seglength ' + str(segment_length) + ' --cutoff ' + str(cutoff) + ' --load-threshold ' + threshold_file + ' -v')

# filtering
os.system(bin_folder + 'cvpr14_filter.py hci --subset test --indir ' + motion_dir + ' --outdir ' + filtered_dir + ' --lambda ' + str(Lambda) + ' --window ' + str(window) + ' --order ' + str(order) + ' -v')

# computing heart-rate
os.system(bin_folder + 'rppg_get_heart_rate.py hci --subset test --indir ' + filtered_dir + ' --outdir ' + hr_dir + ' --framerate ' + str(framerate) + ' --nsegments ' +str(n_segments) + ' --nfft ' + str(nfft) + ' -v')

# computing performance
os.system(bin_folder + 'rppg_compute_performance.py  hci --subset test --indir ' + hr_dir + ' --outdir ' + results_dir_test + ' -v')
