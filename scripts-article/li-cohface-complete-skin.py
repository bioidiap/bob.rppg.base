import os, sys
from gridtk.sge import JobManagerSGE

# directories and file
base_expe_dir = 'experiments/paper/li-cohface-complete-skin/'
facedir = base_expe_dir + 'face'
bgdir = base_expe_dir + 'bg'
illumination_dir = base_expe_dir + 'illumination'
threshold_file = illumination_dir + '/threshold.txt'
motion_dir = base_expe_dir + 'motion'
filtered_dir = base_expe_dir + 'filtered'
hr_dir = base_expe_dir + 'hr'
results_dir_train = base_expe_dir + 'results-train'
results_dir_test = base_expe_dir + 'results-test'

framerate = 20
number_of_jobs = 160 

#parameters
skin_threshold = 0.8

adaptation = 0.01
filter_length = 3

segment_length = 40
cutoff = 0.1

Lambda = 100
window = 3
order = 128

n_segments = 8
nfft = 512

# write a file with the parameters - useful to keep track sometimes ..
param_file = base_expe_dir + '/parameters.txt'
if not os.path.isdir(base_expe_dir):
  os.mkdir(base_expe_dir)

f = open(param_file, 'w')
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

# check job status on the grid
def check_job_status(job_id):
  manager.lock()
  job = manager.get_jobs([job_id])
  manager.unlock()
  return str(job[0].status)

# signals extraction
#manager = JobManagerSGE(database='submitted.sql3', wrapper_script=os.path.abspath('./bin/jman'))
#extract = ['./bin/cvpr14_video2skin.py', 'cohface', '--dbdir', 'cohface', '--bboxdir', 'bounding-boxes', '--outdir', str(facedir), '--threshold', str(skin_threshold)]
#job_id = manager.submit(extract, array=(1, number_of_jobs, 1))
#print 'Running job {0} (extraction) on the grid ...'.format(job_id)
#
## while the job is not succesfull, just wait
#job_status = check_job_status(job_id)
#while (job_status != 'success'):
#  job_status = check_job_status(job_id)

os.system('./bin/cvpr14_video2skin.py cohface --dbdir cohface --bboxdir bounding-boxes --outdir ' + str(facedir) + ' --threshold ' + str(skin_threshold) + ' -v')

# illumination correction
os.system('./bin/cvpr14_illumination.py cohface --facedir ' + facedir + ' --bgdir ' + bgdir + ' --outdir ' + illumination_dir + ' --step ' + str(adaptation) + ' --length ' + str(filter_length) + ' -v')

# motion elimination -> determine the threshold
os.system('./bin/cvpr14_motion.py cohface --subset train --subset dev --indir ' + illumination_dir + ' --outdir ' + motion_dir + ' --seglength ' + str(segment_length) + ' --cutoff ' + str(cutoff) + ' --save-threshold ' + threshold_file + ' -v')

# motion elimination -> remove segments
os.system('./bin/cvpr14_motion.py cohface --indir ' + illumination_dir + ' --outdir ' + motion_dir + ' --seglength ' + str(segment_length) + ' --cutoff ' +str(cutoff) + ' --load-threshold ' + threshold_file + ' -v')

# filtering
os.system('./bin/cvpr14_filter.py cohface --indir ' + motion_dir + ' --outdir ' + filtered_dir + ' --lambda ' + str(Lambda) + ' --window ' + str(window) + ' --order ' + str(order) + ' -v')

# computing heart-rate
os.system('./bin/rppg_get_heart_rate.py cohface --indir ' + filtered_dir + ' --outdir ' + hr_dir + ' --framerate ' + str(framerate) + ' --nsegments ' +str(n_segments) + ' --nfft ' + str(nfft) + ' -v')

# computing performance
os.system('./bin/rppg_compute_performance.py  cohface --subset train --subset dev --indir ' + hr_dir + ' --outdir ' + results_dir_train + ' -v')
os.system('./bin/rppg_compute_performance.py  cohface --subset test --indir ' + hr_dir + ' --outdir ' + results_dir_test + ' -v')
