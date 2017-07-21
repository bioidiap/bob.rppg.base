#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Fri 21 Jul 15:21:49 CEST 2017

import os, sys

# directories and file
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(current_folder)
bin_folder = os.path.join(root_folder, 'bin/')

base_expe_dir = os.path.join(root_folder, 'experiments/paper/ssr-cohface-clean-natural/')
pulse_dir = base_expe_dir + 'pulse'
hr_dir = base_expe_dir + 'hr'
results_dir_train = base_expe_dir + 'results-train'
results_dir_test = base_expe_dir + 'results-test'

framerate = 20

# parameters
skin_threshold = 0.8
stride = 80 

n_segments = 8
nfft = 2048

# write a file with the parameters - useful to keep track sometimes ..
param_file = base_expe_dir + '/parameters.txt'
if not os.path.isdir(base_expe_dir):
  os.makedirs(base_expe_dir)

f = open(param_file, 'w')
f.write('skin threshold = ' + str(skin_threshold) + '\n\n')
f.write('stride = ' + str(stride) + '\n\n')
f.write('Welch segments = ' + str(n_segments) + '\n')
f.write('npoints FFT = ' + str(nfft) + '\n')
f.close()

# pulse extraction
os.system(bin_folder + 'ssr_pulse.py cohface --protocol natural --subset test --dbdir cohface --outdir ' + str(pulse_dir) + ' --threshold ' + str(skin_threshold) + ' --stride ' + str(stride) + ' -v')

# computing heart-rate
os.system(bin_folder + 'rppg_get_heart_rate.py cohface --protocol natural --subset test --indir ' + pulse_dir + ' --outdir ' + hr_dir + ' --framerate ' + str(framerate) + ' --nsegments ' + str(n_segments) + ' --nfft ' + str(nfft) + ' -v')

# computing performance
os.system(bin_folder + 'rppg_compute_performance.py  cohface --protocol natural --subset test --indir ' + hr_dir + ' --outdir ' + results_dir_test + ' -v')
