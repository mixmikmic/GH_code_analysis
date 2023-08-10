import os
import numpy as np
from nibabel import trackvis as tv
from utilities import *
import nibabel as nib
from dipy.tracking import utils

path_saveing = '/data/hcp/data/'
path = '/hcp/'

subjects = os.listdir(path_saveing)
subjects_sorted = sorted(subjects)
subjects_sorted.remove('.nii.gz')

for subject in subjects_sorted:
    print 'Process subject ' + subject

    if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_Strength_of_Evidence3.txt')) == False:
        print "    Strength of Evidence does not exist, start calculation"
        
        if os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_neighborhood.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_with_path.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_without_path.nii.gz')) == True and os.path.isfile(os.path.join(path, subject, 'T1w/Diffusion/data.nii.gz')) == True:
            print "    All necessary files exist, continue"
            
            directory_output = os.path.join(path_saveing, subject)

            mask = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_neighborhood")
            mask = mask.astype(bool)

            with_path = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_with_path")
            without_path = load_nifti(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_rmse_without_path")

            data, affine, gtab, header, shell_mask = load_hcp_data(path, subject)

            noise = data[..., gtab.b0s_mask].std(axis=3, ddof=1) 

            print "    Total Error divided by STD before adding: %.2f" % np.sum(without_path[mask]/noise[mask])
            print "    Total Error divided by STD after adding : %.2f" % np.sum(with_path[mask]/noise[mask])
            strength = np.sum(without_path[mask]/noise[mask]) - np.sum(with_path[mask]/noise[mask])
            print "    Strength of Evidence                    : %.2f" % strength

            np.savetxt(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_Strength_of_Evidence2.txt'), np.array([strength]), delimiter=',')

            print "    Calculate optimized Fibers"
            beta_with = load_matlab_file(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_LiFE_betas_with_path")['beta']

            streams, hdr = tv.read(path_saveing + subject + "/RVLPFC2FIRSTamyg_bigRight_clustered.trk")#, points_space='voxel'
            streamlines = [i[0] for i in streams]

            weights_with = np.squeeze(beta_with[:, -len(streamlines):])
            optimized_with = streamlines[-len(streamlines):]
            
            try:
                optimized_sl = list(np.array(optimized_with)[np.where(weights_with > 0)])

                optimized_save = ((sl, None, None) for sl in optimized_sl)
                tv.write(os.path.join(path_saveing, subject, 'RVLPFC2FIRSTamyg_bigRight_optimized.trk'), optimized_save,  hdr_mapping=hdr)
            except:
                print "Could not save streamlines, 0 streamlines detected"
        else:
            print "    Not all necessary files exist, skip subject"
    else:
        print "Strength of Evidence Exists already, skip subject"



