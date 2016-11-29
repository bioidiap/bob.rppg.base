import os, sys
from gridtk.sge import JobManagerSGE

# directories and file
base_expe_dir = 'experiments/paper/li-hci-cvpr14/'
facedir = base_expe_dir + 'face'
bgdir = base_expe_dir + 'bg'
illumination_dir = base_expe_dir + 'illumination'
threshold_file = illumination_dir + '/threshold.txt'
motion_dir = base_expe_dir + 'motion'
filtered_dir = base_expe_dir + 'filtered'
hr_dir = base_expe_dir + 'hr'
results_dir = base_expe_dir + 'results'

# fixed parameter
framerate = 61

# parameters
npoints = 200
indent = 10 

adaptation = 0.05
filter_length = 3

segment_length = 61
cutoff = 0.05

Lambda = 300
window = 21
order = 128

n_segments = 16
nfft = 8192

# write a file with the parameters - useful to keep track sometimes ..
param_file = base_expe_dir + '/parameters.txt'
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


# check job status on the grid
def check_job_status(job_id):
  manager.lock()
  job = manager.get_jobs([job_id])
  manager.unlock()
  return str(job[0].status)

# job manager on the grid
manager = JobManagerSGE(database='submitted.sql3', wrapper_script=os.path.abspath('./bin/jman'))

# signals extraction
extract = ['./bin/cvpr14_extract_signals.py', 'hci',  '--protocol',  'cvpr14',  '--dbdir', 'hci', '--bboxdir', 'bounding-boxes', '--facedir', str(facedir), '--bgdir', str(bgdir), '--npoints', str(npoints), '--indent', str(indent)]
job_id = manager.submit(extract, array=(1, 527, 1))
print 'Running job {0} (extraction) on the grid ...'.format(job_id)

# while the job is not succesfull, just wait
job_status = check_job_status(job_id)
while (job_status != 'success'):
  job_status = check_job_status(job_id)

# illumination correction
os.system('./bin/cvpr14_illumination.py hci --protocol cvpr14 --facedir ' + facedir + ' --bgdir ' + bgdir + ' --outdir ' + illumination_dir + ' --step ' + str(adaptation) + ' --length ' + str(filter_length) + ' --start 306 --end 2136 -v')

# motion elimination -> determine the threshold
os.system('./bin/cvpr14_motion.py hci --protocol cvpr14 --indir ' + illumination_dir + ' --outdir ' + motion_dir + ' --seglength ' + str(segment_length) + ' --cutoff ' + str(cutoff) + ' --save-threshold ' + threshold_file + ' -v')

# motion elimination -> remove segments
os.system('./bin/cvpr14_motion.py hci --protocol cvpr14 --indir ' + illumination_dir + ' --outdir ' + motion_dir + ' --seglength 61 --cutoff 0.05 --load-threshold ' + threshold_file + ' -v')

# filter
os.system('./bin/cvpr14_filter.py hci --protocol cvpr14 --indir ' + motion_dir + ' --outdir ' + filtered_dir + ' --lambda ' + str(Lambda) + ' --window ' + str(window) + ' --order ' + str(order) + ' -v')

# computing heart-rate
os.system('./bin/rppg_get_heart_rate.py hci --protocol cvpr14 --indir ' + filtered_dir + ' --outdir ' + hr_dir + ' --framerate ' + str(framerate) + ' --nsegments ' +str(n_segments) + ' --nfft ' + str(nfft) + ' -v')

# computing performance
os.system('./bin/rppg_compute_performance.py  hci --protocol cvpr14 --indir ' + hr_dir + ' --outdir ' + results_dir + ' -v')
