import os
import glob
import subprocess
import ipyparallel as ipp
import shutil

def run_fvs(keyfile):
    fvs_exe = 'C:\\FVSbin\\'+os.path.split(keyfile)[-1][:5]+'.exe'
    subprocess.call([fvs_exe, '--keywordfile='+keyfile]) # run fvs
    
    base_dir = os.path.split(keyfile)[0]
    base_name = os.path.split(keyfile)[-1].split('.')[0]
    
    # clean-up the outputs
    # move the .out and .key file
    path = os.path.join(base_dir, 'completed','keyfiles')
    if not os.path.exists(path): 
        os.makedirs(path)
    shutil.move(keyfile, os.path.join(base_dir,'completed','keyfiles'))
    path = os.path.join(base_dir, 'completed','outfiles')
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.move(os.path.join(base_dir,base_name+'.out'), os.path.join(base_dir,'completed','outfiles'))
    
    # delete the other files
    os.remove(os.path.join(base_dir, base_name+'.trl'))
    return keyfile

# create a hub to control the workers
c = ipp.Client()
c.ids

# if you want to run a single keyfile, use this
# subprocess.call(['C:\\FVSbin\\FVSpn.exe', '--keywordfile=C:\\GitHub\\FSC_Case_Studies\\keyfiles_to_run\\PN\\fvsPN_stand1_rx4_off0.key'])

dv = c[:] # direct view
v = c.load_balanced_view() # load-balanced view

# import packages to all workers
with dv.sync_imports():
    import subprocess
    import shutil
    import os

# gather the list of keyfiles to run
run_dir = os.path.abspath('keyfiles_to_run')
to_run = glob.glob(os.path.join(run_dir, '*.key'))
print('{:,}'.format(len(to_run)), 'keyfiles found.')

# start asynchronous batch with load-balanced view
res = v.map_async(run_fvs, to_run)
print('Started batch processing.')

# Default method
# res.wait_interactive()

# OR USE A PROGRESS BAR!
from tqdm import tqdm_notebook
import time

runs_done = res.progress
with tqdm_notebook(total=len(res), initial=runs_done, desc='FVS Run Progress', unit='keyfile') as pbar:
    new_progress = res.progress - runs_done
    runs_done += new_progress
    pbar.update(new_progress)

# Return a true/false if full set of jobs completed
# res.ready()

# Cancels the batch (wait for fvs executables to complete)
# res.abort()

print('Human time spent:', res.wall_time)
print('Computer time spent:', res.serial_time)
print('Async speedup:', res.serial_time/res.wall_time)
print('Human time per FVS run:', res.wall_time/res.progress)
print('Computer time per FVS run:', res.serial_time/res.progress)

# inspect how processing speed per run changed as batch progressed
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
time_steps = [(t2 - t1).total_seconds() for t2, t1 in zip(res.received, res.submitted)]
plt.plot(time_steps)
plt.ylabel('time per run')
plt.xlabel('runs completed')
plt.show()

# shut down the parallel workers
c.shutdown(hub=True)

# import psycopg2
# import pandas as pd
# conn = psycopg2.connect("dbname='FVSOut' user='postgres' host='localhost'") # password in pgpass file
# SQL = '''
# SELECT keywordfile
# FROM fvs_cases;
# '''
# # read the query into a pandas dataframe
# completed = pd.read_sql(SQL, conn)

# # close the database connection
# conn.close()

# completed['keyfile'] = completed.keywordfile.apply(lambda x: os.path.split(x)[-1] + '.key')
# completed.keyfile.values[0]

# completed_keys = glob.glob('C:\\GitHub\\FSC_Case_Studies\\keyfiles_to_run\\PN\\completed\\keyfiles\\*.key')
# completed_basenames = [os.path.split(x)[-1] for x in completed_keys]
# print(len(completed), 'keyfiles in database')
# print(len(completed_basenames), 'keyfiles in completed folder')

# for keyfile in completed.keyfile.values: # keyfiles recorded in the DB
#     if keyfile not in completed_basenames:
#         print(keyfile)

# failed = glob.glob('C:\\GitHub\\FSC_Case_Studies\\keyfiles_to_run\\PN\\completed\\outfiles\\failed\\*.out')
# failed_basenames = [os.path.split(x)[-1].split('.')[0] for x in failed]
# moved = glob.glob('C:\\GitHub\\FSC_Case_Studies\\keyfiles_to_run\\*.key')
# moved_basenames = [os.path.split(x)[-1].split('.')[0] for x in moved]
# for path in moved:
#     if os.path.split(path)[-1].split('.')[0] not in failed_basenames:
#         print(path, "not in failed, but was moved")
# for path in failed:
#     if os.path.split(path)[-1].split('.')[0] not in moved_basenames:
#         print(path, "not in moved, but failed")

