import os
import re
import copy
import numpy as np
import pandas as pd
import nibabel as nib
import brainbox as bb
import multiprocessing as mp
from matplotlib import pyplot as plt

import os
import re
import glob
import copy
import numpy as np
import nibabel as nib
from brainbox import tools as to
from  __builtin__ import any as b_any

def find_files(in_path, ext, targets, template='(?<=\d{2})\d{5}', sub=False):
    """
    Finds matching files with extension ext and returns them in
    the order of the targets list given as argument
    Returns a dictionary identical to what I was using before
    Also drops duplicates
    """
    # Go through each directory and see if I can find the subjects I am looking
    # for
    ext = '*{}'.format(ext)
    out_dict = {key: [] for key in ['sub_name', 'dir', 'path']}
   
    if not sub:
        sub_dirs = [d for d in os.walk(in_path).next()[1]]
        print(sub_dirs)
        for sub_dir in sub_dirs:
            print('heyho')
            tmp_dir = os.path.join(in_path, sub_dir)
            in_files = glob.glob(os.path.join(tmp_dir, ext))
            tmp_dict = dict()

            # Get the files that we have
            matches = [x for x in targets if b_any(str(x) in t for t in in_files)]

            for in_file in in_files:
                sub_name = os.path.basename(in_file.split('.')[0])
                sub_id = re.search(r'{}'.format(template), sub_name).group()
                if sub_id in tmp_dict.keys():
                    # This is a duplicate
                    continue
                tmp_dict[sub_id] = (sub_name, in_file)

            # Re-sort the path info
            sort_list = list()
            for target in matches:
                sub_name, in_file = tmp_dict[target]
                out_dict['sub_name'].append(sub_name)
                out_dict['dir'].append(sub_dir)
                out_dict['path'].append(in_file)
    else:
        sub_dir = sub
        tmp_dir = os.path.join(in_path, sub_dir)
        in_files = glob.glob(os.path.join(tmp_dir, ext))
        tmp_dict = dict()

        # Get the files that we have
        matches = [x for x in targets if b_any(str(x) in t for t in in_files)]

        for in_file in in_files:
            sub_name = os.path.basename(in_file.split('.')[0])
            sub_id = re.search(r'{}'.format(template), sub_name).group()
            if sub_id in tmp_dict.keys():
                # This is a duplicate
                continue
            tmp_dict[sub_id] = (sub_name, in_file)

        for target in matches:
            sub_name, in_file = tmp_dict[target]
            out_dict['sub_name'].append(sub_name)
            out_dict['dir'].append(sub_dir)
            out_dict['path'].append(in_file)
    return out_dict

# Paths
scale = 7
run = 1
method = 'stability_maps'
directory = '{}_sc{}'.format(method, scale)
in_path = '/data1/pierre_scores/rest_{}'.format(run)
out_path = '/data1/pierre_scores/out/scores_s{}/stack_maps_{}'.format(scale, run)
if not os.path.isdir(out_path):
    try:
        os.makedirs(out_path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_path):
            pass
        else: raise

ta_path = '/data1/pierre_scores/pheno/model_preventad_20141215.csv'
ext = '.nii.gz'

pheno = pd.read_csv(ta_path)
targets = pheno['subjects'].values
file_dict = find_files(in_path, ext, targets, template='(?<=fmri_)s\d{6}\S{3}', sub=directory)
num_subs = len(file_dict['path'])
data_subs = np.array([re.search(r'(?<=fmri_)s\d{6}\S{3}', sub_id).group() for sub_id in file_dict['sub_name']])

def run_par(args):
    """
    Wrapper function to do the loading and saving in parallel
    """
    ds, num_subs, use_dict, tmp_i, out_path, net_id = args
    
    mean_mat = np.zeros(ds[:3] + (scale,))
    std_mat = np.zeros(ds[:3] + (scale,))
    
    sub_stack = np.zeros(ds[:3] + (num_subs,))
    for sub_id in np.arange(num_subs):
        img = nib.load(use_dict['path'][sub_id])
        data = img.get_data()
        net = data[..., net_id]
        sub_stack[..., sub_id] = net
    # Save the network stack first
    stack_out = nib.Nifti1Image(sub_stack, tmp_i.get_affine(), tmp_i.get_header())
    nib.save(stack_out, os.path.join(out_path, '{}_netstack_net{}_scale_{}_run_{}.nii.gz'.format(method, net_id + 1, scale, run)))
    # Get the mean and std
    mean = np.mean(sub_stack, axis=3)
    mean_mat[..., net_id] = mean
    std = np.std(sub_stack, axis=3)
    std_mat[..., net_id] = std
    
    return mean_mat, std_mat

# Get a template
tmp_i = nib.load(file_dict['path'][0])
tmp = tmp_i.get_data()
ds = tmp.shape

# Set up the parallel processing
p_perc = 0.9
p_count = int(np.floor(mp.cpu_count() * p_perc))

# Prepare the meta mats
mean_mat = np.zeros(ds[:3] + (scale,))
std_mat = np.zeros(ds[:3] + (scale,))
arg_list = list()
for net_id in np.arange(scale):
    arg_list.append((ds, num_subs, file_dict, tmp_i, out_path, net_id))
    
# Run the stuff in parallel
print('Running things in parallel now - for speed!')
p = mp.Pool(p_count)
results = p.map(run_par, arg_list)
print('Done with that')

for result in results:
    mean_mat += result[0]
    std_mat += result[1]

# Save the mean and std maps
print('Saving the mean and std files to {}'.format(out_path))
mean_out = nib.Nifti1Image(mean_mat, tmp_i.get_affine(), tmp_i.get_header())
nib.save(mean_out, os.path.join(out_path, '{}_mean_stack_scale{}_{}.nii.gz'.format(method, scale, run)))
std_out = nib.Nifti1Image(std_mat, tmp_i.get_affine(), tmp_i.get_header())
nib.save(std_out, os.path.join(out_path, '{}_std_stack_scale{}_run_{}.nii.gz'.format(method, scale, run)))



