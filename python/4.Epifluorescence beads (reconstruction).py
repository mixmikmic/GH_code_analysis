get_ipython().magic('matplotlib inline')
import sys, os
import numpy as np
from libtiff import TIFF
import matplotlib.pyplot as plt
sys.path.append("../../3_code/")
import cstools
reload(cstools)

# useful function
def spiral(m, b):
    """Poisson likelihood objective function"""
    return cstools.reconstruct_1Dspiral(m, b, maxiter=1000, noisetype='poisson')
def spiral2(m, b):
    """Gaussian likelihood objective function"""
    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000)
def spiral_tv(m, b): 
    """Total variation reconstruction"""
    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000, penalty='tv')

# Define paths
bn = "../../4_data/epifluorescence/161010_CS_beads_mega/"
p_psf = os.path.join(bn, "3_eye_3/Pos0_eye_PSF.tif")
p_eye = os.path.join(bn, "3_eye_3/Pos0_eye.tif")
p_fou = os.path.join(bn, "3_fourier_2/Pos0_cs.tif")

# load data
psf_f = cstools.read_tif(p_psf)
eye = cstools.read_tif(p_eye)
fou = cstools.read_tif(p_fou)

# Measurement matrix
b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)

## 1. Extract a PSF
psfz_ok = True # Flag to switch between calibration and generation of the PSF
psfx_ok = True
save_psf = "../../5_outputs/psf_models/epifluorescence_161121" ## Set to None to avoid saving
load_psf = False ## Load the PSF from the save_psf file.
step = 20 ## A fraction of the size of the PSF in z. Represent the shift in the final dictionary.

psf = psf_f[15:25,:,15:25].mean(0)
psf -= psf.min()

if save_psf != None and not load_psf and not os.path.isfile(save_psf): ## Save the PSF if needed
    np.save(save_psf, psf)
    print "PSF saved on file {}".format(save_psf)
elif not load_psf and save_psf != None and os.path.isfile(save_psf):
    raise IOError("File {} exists".format(save_psf))    
else:
    print "Not saving the PSF"

if load_psf:
    psf=np.load(save_psf+".npy")
    print "Loaded PSF from file {}.npy".format(save_psf)


l=np.zeros((eye.shape[2], 2*psf.shape[1]))
l[:psf.shape[0],:psf.shape[1]]=psf
ll=[]
for j in range(step*int(eye.shape[2]/psf.shape[0])+1):
    for k in range(2*psf.shape[0]):
        ll.append(np.roll(np.roll(l, psf.shape[0]*psf.shape[1]*j/step, axis=0), k, axis=1)[:,int(l.shape[1]/2):int(l.shape[1]*3./2)])

D1 = np.hstack(ll) ## This is the dictionary.
D2 = np.hstack([i.T.reshape((-1,1)) for i in ll])

print D1.shape, D2.shape
plt.figure(figsize=(18,5))
plt.imshow(D2, cmap='gray', interpolation="none")

B = cstools.build_2dmodel(b, psf.mean(0)).dot(D2)

## Apply the 2D model
# ==== 3. Subdivide the simulated image & reconstruct (with sliding windows)
def reco_image2(ima, verbose=True):
    im = np.zeros((ima.shape[0], b.shape[1]))
    vec_counts = np.zeros(ima.shape[0])
    for i in range(eye[0, :,:].shape[0]):
        if verbose and i%100 == 0:
            print "-- {}/{}".format(i, eye[0, :,:].shape[0])
        (sta,sto) = (i-psf.shape[1], i)
        im_sub = ima[range(sta,sto),:].flatten()
        r_sim = spiral2(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))        
        im[range(sta,sto),:]+=r_sim
        vec_counts[range(sta,sto)]+=1
        
    for (i,v) in enumerate(vec_counts): # Average each row
        im[i,:] /= v
    return im

im = reco_image2(fou[205,:,:])

plt.figure(figsize=(18,6))
plt.imshow(im.T, cmap='gray', interpolation='none')

## /!\ Be careful that this might take a lot of time
## Basically, this takes 512 times more time than the previous iteration
## /!\ ONCE AGAIN, 3'x512 = 1500' = ONE DAY
## A cluster version is provided below, consider it before doing weird things.

# Launch a full stack reconstruction
im3d = []
for i in range(fou.shape[0]):
    if i%10 == 0:
        print "{}/{}".format(i, fou.shape[0])
    im = reco_image2(fou[i,:,:], verbose=False)
    im3d.append(im)

iff = TIFF.open("./img/paper4.epifluorescence_rec_avg.tif", "w")
iff.write_image([(101*j).astype(np.uint16) for j in im3d])
iff.close()

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/08_epifluorescence/reconstruction4.sh', '#!/bin/sh\n## /!\\ This code is generated by a Jupyter script, do not edit it directly.\n## It is designed to run reconstructions on a very specific dataset.\n## It should be adapted carefully before playing on the input stuff\n\necho "==== DISCLAIMER ===="\necho "Have you installed the following packages?: virtualenv, numpy, scipy libtiff, joblib, pycsalgos, h5py, pySPIRALTAP"\necho "Have you run the following commands to load the packages?"\necho "$ module load Python/2.7.11"\necho "$ source ~/.local/bin/virtualenvwrapper.sh"\necho "$ export WORKON_HOME=~/.envs"\necho "$ workon dict-optim"\n\nN_FRAMES_IN_STACK=512\nemail="maxime.woringer@pasteur.fr"\nmaxparalleljobs=100\n\necho \'Running on tars\'\nsbatch --mail-type=BEGIN,END --mail-user=$email --array=0-$N_FRAMES_IN_STACK%$maxparalleljobs ../8_cluster/tars/08_epifluorescence/reconstruction4_init.sh')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/08_epifluorescence/reconstruction4_init.sh', '#!/bin/sh\n#SBATCH --qos=fast \n#SBATCH -N 1\n#SBATCH -c 12\n#SBATCH -p common,dedicated\n#SBATCH -o ../8_cluster/tars/06_reconstruct_3d/log_reconstruction.log -e ../8_cluster/tars/06_reconstruct_3d/log_reconstruction.err\n# By Maxime W., Nov. 2016, GPLv3+\n# /!\\ DO NOT EDIT THIS FILE. IT HAS BEEN GENERATED BY A SCRIPT\n# Script is ../2_simulations/51. ...ipynb\n\n## This script to be called by SBATCH, do not call it directly, it will not work.\nsrun ~/.envs/dict-optim/bin/python ../8_cluster/tars/08_epifluorescence/reconstruction4.py ${SLURM_ARRAY_TASK_ID}')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/08_epifluorescence/reconstruction4.py', '\nimport sys, os\nimport numpy as np\nfrom libtiff import TIFF\nsys.path.append("../3_code/")\nimport cstools\n#reload(cstools)\n\nn_mes = 50 ## Compression factor (a number of frames)\nsave_prefix = "epifluorescence{}".format(n_mes)\n\n\n## tars-specific parameters\nframe_to_reconstruct = int(sys.argv[1])\n\n\n# useful function\ndef spiral(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=1000, noisetype=\'poisson\')\ndef spiral2(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000)\n\n# Define paths\nbn = "../4_data/epifluorescence/161010_CS_beads_mega/"\np_psf = os.path.join(bn, "3_eye_3/Pos0_eye_PSF.tif")\np_eye = os.path.join(bn, "3_eye_3/Pos0_eye.tif")\np_fou = os.path.join(bn, "3_fourier_2/Pos0_cs.tif")\n\nout_folder = "../5_outputs/11_reconstructions_march_epifluorescence"\n\n## Do not reconstruct if we already have it\nif os.path.exists(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct))):\n    print "reconstruction {} exists, aborting".format("{}.{}.txt".format(save_prefix, frame_to_reconstruct))\n    sys.exit(0)\n\n# load data\npsf_f = cstools.read_tif(p_psf)\neye = cstools.read_tif(p_eye)\nfou = cstools.read_tif(p_fou)\n\n###\n### ===================== Extract a PSF and build measurement matrix\n###\npsfz_ok = True # Flag to switch between calibration and generation of the PSF\npsfx_ok = True\nsave_psf = "../5_outputs/psf_models/epifluorescence_161121" ## Set to None to avoid saving\nload_psf = False ## Load the PSF from the save_psf file.\nstep = 20 ## A fraction of the size of the PSF in z. Represent the shift in the final dictionary.\n\npsf = psf_f[15:25,:,15:25].mean(0)\npsf -= psf.min()\n\nif save_psf != None and not load_psf and not os.path.isfile(save_psf): ## Save the PSF if needed\n    np.save(save_psf, psf)\n    print "PSF saved on file {}".format(save_psf)\nelif not load_psf and save_psf != None and os.path.isfile(save_psf):\n    raise IOError("File {} exists".format(save_psf))    \nelse:\n    print "Not saving the PSF"\n\nif load_psf:\n    psf=np.load(save_psf+".npy")\n    print "Loaded PSF from file {}.npy".format(save_psf)\n\n\nl=np.zeros((eye.shape[2], 2*psf.shape[1]))\nl[:psf.shape[0],:psf.shape[1]]=psf\nll=[]\nfor j in range(step*int(eye.shape[2]/psf.shape[0])+1):\n    for k in range(2*psf.shape[0]):\n        ll.append(np.roll(np.roll(l, psf.shape[0]*psf.shape[1]*j/step, axis=0), k, axis=1)[:,int(l.shape[1]/2):int(l.shape[1]*3./2)])\n\nD1 = np.hstack(ll) ## This is the dictionary.\nD2 = np.hstack([i.T.reshape((-1,1)) for i in ll])\n    \n###\n### ======================== Generate stuff\n###\n\n## Generate measurement matrix\n# Generate basis\nif n_mes == 0:\n    b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)\nelse:\n    print "Truncating"\n    b=cstools.generate_fourier_basis(101, n_mes+1, sample=False, oldmethod=True)\n    fou = fou[:,:,:n_mes]\n\nB = cstools.build_2dmodel(b, psf.mean(0)).dot(D2)\n\n## Apply the 2D model\n# ==== 3. Subdivide the simulated image & reconstruct (with sliding windows)\ndef reco_image(ima, verbose=True):\n    im = np.zeros((ima.shape[0], b.shape[1]))\n    vec_counts = np.zeros(ima.shape[0])\n    for i in range(eye[0, :,:].shape[0]):\n        if verbose and i%100 == 0:\n            print "-- {}/{}".format(i, eye[0, :,:].shape[0])\n        (sta,sto) = (i-psf.shape[1], i)\n        im_sub = ima[range(sta,sto),:].flatten()\n        r_sim = spiral2(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))        \n        im[range(sta,sto),:]+=r_sim\n        vec_counts[range(sta,sto)]+=1\n        \n    for (i,v) in enumerate(vec_counts): # Average each row\n        im[i,:] /= v\n    return im\n###\n### =========================== Reconstruction step\n###\nprint "Reconstructing"\nim = reco_image(fou[frame_to_reconstruct,:,:], verbose=False) # Launch a full stack reconstruction\n\n###\n### =========================== Saving step\n###\nnp.savetxt(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct)), im)')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/08_epifluorescence/recombine.py', '## Maxime W., Jul 2016, GPLv3+\n## This is actually the worker to combine the movie to a 3D stack\n\n# ==== Imports\nimport sys, os\nimport numpy as np\nfrom libtiff import TIFF\n\ndim1 = 512\ndim2 = 101\nscl = 1000 # Multiply by this factor before saving.\n\n\nprint "Saving a stack of {} elements, {} px".format(dim2, dim1)\n\n# ==== Variables\n(none, frame_bn, out_dir) = sys.argv\ntmp_dir = os.path.join(out_dir, \'\')\nout_file = os.path.join(out_dir, "{}.tif".format(frame_bn))\nremove_neg = True # if True, negative elements in the reconstruction will be set to zero.\n\n# ==== Checking inputs\n## Input files\ninp_files = [i for i in os.listdir(tmp_dir) if i.startswith(frame_bn)]\nif inp_files == []:\n    raise IOError(\'No files found in {} with name starting with {}. ABORTING\'.format(tmp_dir, frame_bn))\n    \n## Output file\nif os.path.isfile(out_file):\n    im = TIFF.open(out_file, \'r\')\n    for i in im.iter_images():\n        frame0=i\n        break\n    im.close()\n    if frame0.shape[0] >= len(inp_files):\n        print "A TIFF stack already exists and has at least as many pixels as what we"\n        print "were aiming to reconstruct. EXITING."\n        sys.exit(0)\n\n## ==== Saving image\nprint(\'Image dimensions are hardcoded so far. I know this is stupid.\')\nidx = [int(i.split(\'.\')[-2]) for i in inp_files]\nlf = sorted(zip(inp_files, idx), key=lambda x: x[1])\nim = np.zeros((len(lf), dim1, dim2))\nfor (i,j) in enumerate(lf):\n    f=np.genfromtxt(os.path.join(tmp_dir, j[0]))\n    im[i,:,:]=f.T\nprint "Loaded {} planes".format(len(lf))\n\nif remove_neg:\n    print "Negative elements were set to zero"\n    im[im<0]=0 ## Bruteforce.\ntif = TIFF.open(out_file, \'w\')\ntif.write_image(np.array(np.int_(im*scl).swapaxes(2, 0).swapaxes(1,2), dtype=np.uint16))\ntif.close()')

cstools.read_tif
