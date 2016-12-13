import os, sys
from gridtk.sge import JobManagerSGE

# directories and file
base_expe_dir = 'experiments/paper/ssr-cohface-complete-skin/'
pulse_dir = base_expe_dir + 'pulse'
hr_dir = base_expe_dir + 'hr'
results_dir_train = base_expe_dir + 'results-train'
results_dir_test = base_expe_dir + 'results-test'

number_of_jobs = 160
framerate = 20

# parameters
skin_threshold = 0.6
stride = 10

n_segments = 8
nfft = 512

# write a file with the parameters - useful to keep track sometimes ..
param_file = base_expe_dir + '/parameters.txt'
if not os.path.isdir(base_expe_dir):
  os.mkdir(base_expe_dir)

f = open(param_file, 'w')
f.write('threshold = ' + str(skin_threshold) + '\n\n')
f.write('stride = ' + str(stride) + '\n\n')
f.write('Welch segments = ' + str(n_segments) + '\n')
f.write('npoints FFT = ' + str(nfft) + '\n')
f.close()
f.close()

# check job status on the grid
def check_job_status(job_id):
  manager.lock()
  job = manager.get_jobs([job_id])
  manager.unlock()
  return str(job[0].status)

# pulse extraction
manager = JobManagerSGE(database='submitted.sql3', wrapper_script=os.path.abspath('./bin/jman'))
extract = ['./bin/ssr_pulse.py', 'cohface', '--dbdir', 'cohface', '--bboxdir', 'bounding-boxes', '--outdir', str(pulse_dir), '--threshold', str(skin_threshold), '--stride', str(stride)]
job_id = manager.submit(extract, array=(1, number_of_jobs, 1))
print 'Running job {0} (extraction) on the grid ...'.format(job_id)

# while the job is not succesfull, just wait
job_status = check_job_status(job_id)
while (job_status != 'success'):
  job_status = check_job_status(job_id)

os.system('./bin/ssr_pulse.py cohface --dbdir cohface --bboxdir bounding-boxes --outdir ' + str(pulse_dir) + ' --threshold ' + str(skin_threshold) + ' --stride ' + str(stride) + ' -v')

# computing heart-rate
os.system('./bin/rppg_get_heart_rate.py cohface --indir ' + pulse_dir + ' --outdir ' + hr_dir + ' --framerate ' + str(framerate) + ' --nsegments ' +str(n_segments) + ' --nfft ' + str(nfft) + ' -v')

# computing performance
os.system('./bin/rppg_compute_performance.py cohface --subset train --subset dev --indir ' + hr_dir + ' --outdir ' + results_dir_train + ' -v')
os.system('./bin/rppg_compute_performance.py cohface --subset test --indir ' + hr_dir + ' --outdir ' + results_dir_test + ' -v')
