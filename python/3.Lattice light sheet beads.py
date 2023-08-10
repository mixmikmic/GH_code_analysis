get_ipython().magic('matplotlib inline')
import sys, os
import scipy.linalg
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

bn = "../../4_data/lattice-lightsheet/161013_cstest/"
eye_p = os.path.join(bn, "eyefullfov20um.tif")
fou_p = os.path.join(bn, "fourierfullfov20um.tif")

# Load matrices
fou = cstools.read_tif(fou_p, div2=True)
eye = cstools.read_tif(eye_p, div2=True)

## 1. Extract a PSF
psfz_ok = True # Flag to switch between calibration and generation of the PSF
psfx_ok = True
save_psf = "../../5_outputs/psf_models/lattice_161114" ## Set to None to avoid saving
load_psf = True ## Load the PSF from the save_psf file.

step = 8 ## A fraction of the size of the PSF in z. Represent the shift in the final dictionary.
psf_x = 205
psf_y = range(363, 367)
psf_z = (82, 95)

if load_psf:
    psf=np.load(save_psf+".npy")
    print "Loaded PSF from file {}.npy".format(save_psf)
else:
    psf = eye.mean(0)[psf_y,psf_z[0]:psf_z[-1]].T
    psf -= psf.min()

if save_psf != None and not load_psf and not os.path.isfile(save_psf): ## Save the PSF if needed
    np.save(save_psf, psf)
    print "PSF saved on file {}".format(save_psf)
elif not load_psf and save_psf != None and os.path.isfile(save_psf):
    raise IOError("File {} exists".format(save_psf))
else:
    print "Not saving the PSF"

if not psfz_ok:
    plt.imshow(psf, cmap='gray', interpolation="none")
elif not psfx_ok:
    psf_xutil = psf.mean(1)
    plt.plot(psf_xutil)
else:
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

## Generate measurement matrix
b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)
B = cstools.build_2dmodel(b, psf.mean(0)).dot(D2)

## Apply the 2D model
def reco_image(ima, verbose=True, tv=False):
    rr = np.arange(0, eye[0, :,:].shape[0], psf.shape[1])
    im = []
    for (i,j) in enumerate(zip(rr[:-1],rr[1:])):
        if verbose and i%20 == 0:
            print "{}/{}".format(i, len(rr))
        (sta,sto) = j
        #im_sub = fou[ima, range(sta,sto),:].flatten()
        im_sub = ima[range(sta,sto),:].flatten()
        if tv:
            r_sim = spiral_tv(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))
        else:
            r_sim = spiral2(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))
        im.append(r_sim)
    im = np.vstack(im).T
    return im

## Apply the 2D model
def reco_image2(ima, verbose=True, tv=False):
    rr1 = np.arange(0, eye[0, :,:].shape[0]-psf.shape[1], 1)
    rr2 = rr1 + psf.shape[1]
    im = np.zeros_like(eye[0,:,:].T)
    for (i,j) in enumerate(zip(rr1,rr2)):
        if verbose and i%100 == 0:
            print "{}/{}".format(i, len(rr1))
        (sta,sto) = j
        #im_sub = fou[ima, range(sta,sto),:].flatten()
        im_sub = ima[range(sta,sto),:].flatten()
        if tv:
            r_sim = spiral_tv(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))
        else:
            r_sim = spiral2(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))
        #im.append(r_sim)
        im[:,sta:sto]+=r_sim.T
    #im = np.vstack(im).T
    #im = np.zeros(())
    return im

im = reco_image2(fou[205,:,:])
#im_tv = reco_image(fou[205,:,:], tv=True)

plt.figure(figsize=(18,13))
plt.subplot(311);plt.imshow(eye[205,:,:].T, cmap='gray', interpolation="none")
plt.subplot(312);plt.imshow(im, cmap='gray', interpolation="none")
plt.subplot(313);plt.imshow(B, cmap='gray', interpolation="none")

## /!\ Be careful that this might take a lot of time
## Basically, this takes 512 times more time than the previous iteration
## /!\ ONCE AGAIN, 3'x512 = 1500' = ONE DAY
## A cluster version is provided below, consider it before doing weird things.

# Launch a full stack reconstruction
im3d = []
for i in range(fou.shape[0]):
    if i% 10 == 0:
        print "{}/{}".format(i,fou.shape[0])
    im = reco_image(fou[i,:,:], verbose=False)
    im3d.append(im)

iff = TIFF.open("./img/paper3.beads_lattice_avg2D.tif", "w")
iff.write_image([(100*j).astype(np.uint16) for j in im3d])
iff.close()

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/07_reconstruction_3d/reconstruction3.sh', '#!/bin/sh\n## /!\\ This code is generated by a Jupyter script, do not edit it directly.\n## It is designed to run reconstructions on a very specific dataset.\n## It should be adapted carefully before playing on the input stuff\n\necho "==== DISCLAIMER ===="\necho "Have you installed the following packages?: virtualenv, numpy, scipy libtiff, joblib, pycsalgos, h5py, pySPIRALTAP"\necho "Have you run the following commands to load the packages?"\necho "$ module load Python/2.7.11"\necho "$ source ~/.local/bin/virtualenvwrapper.sh"\necho "$ export WORKON_HOME=~/.envs"\necho "$ workon dict-optim"\n\nN_FRAMES_IN_STACK=512\nemail="maxime.woringer@pasteur.fr"\nmaxparalleljobs=100\n\necho \'Running on tars\'\nsbatch --mail-type=BEGIN,END --mail-user=$email --array=0-$N_FRAMES_IN_STACK%$maxparalleljobs ../8_cluster/tars/07_reconstruction_3d/reconstruction3_init.sh')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/07_reconstruction_3d/reconstruction3_init.sh', '#!/bin/sh\n#SBATCH --qos=fast \n#SBATCH -N 1\n#SBATCH -c 12\n#SBATCH -p common,dedicated\n#SBATCH -o ../8_cluster/tars/06_reconstruct_3d/log_reconstruction.log -e ../8_cluster/tars/06_reconstruct_3d/log_reconstruction.err\n# By Maxime W., Nov. 2016, GPLv3+\n# /!\\ DO NOT EDIT THIS FILE. IT HAS BEEN GENERATED BY A SCRIPT\n# Script is ../2_simulations/51. ...ipynb\n\n## This script to be called by SBATCH, do not call it directly, it will not work.\nsrun ~/.envs/dict-optim/bin/python ../8_cluster/tars/07_reconstruction_3d/reconstruction3.py ${SLURM_ARRAY_TASK_ID}')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/07_reconstruction_3d/reconstruction3.py', '\nimport sys, os\nimport scipy.linalg\nimport numpy as np\nfrom libtiff import TIFF\nsys.path.append("../3_code/")\nimport cstools\n#reload(cstools)\n\nn_mes = 30 ## Compression factor (a number of frames)\nsave_prefix = "rec{}".format(n_mes)\n\n\n## tars-specific parameters\nframe_to_reconstruct = int(sys.argv[1])\n\n\n# useful function\ndef spiral(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=1000, noisetype=\'poisson\')\ndef spiral2(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000)\n\nbn = "../4_data/lattice-lightsheet/161013_cstest/"\neye_p = os.path.join(bn, "eyefullfov20um.tif")\nfou_p = os.path.join(bn, "fourierfullfov20um.tif")\n\nout_folder = "../5_outputs/10_reconstructions_march_tars"\n\n## Do not reconstruct if we already have it\nif os.path.exists(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct))):\n    print "reconstruction {} exists, aborting".format("{}.{}.txt".format(save_prefix, frame_to_reconstruct))\n    sys.exit(0)\n\n# Load matrices\nfou = cstools.read_tif(fou_p, div2=True)\neye = cstools.read_tif(eye_p, div2=True)\n\n###\n### ========================================\n###\n\n## 1. Extract a PSF\npsfz_ok = True # Flag to switch between calibration and generation of the PSF\npsfx_ok = True\nsave_psf = "../5_outputs/psf_models/lattice_161114" ## Set to None to avoid saving\nload_psf = True ## Load the PSF from the save_psf file.\n\nstep = 8 ## A fraction of the size of the PSF in z. Represent the shift in the final dictionary.\npsf_x = 205\npsf_y = range(363, 367)\npsf_z = (82, 95)\n\nif load_psf:\n    psf=np.load(save_psf+".npy")\n    print "Loaded PSF from file {}.npy".format(save_psf)\nelse:\n    psf = eye.mean(0)[psf_y,psf_z[0]:psf_z[-1]].T\n    psf -= psf.min()\n\nif save_psf != None and not load_psf and not os.path.isfile(save_psf): ## Save the PSF if needed\n    np.save(save_psf, psf)\n    print "PSF saved on file {}".format(save_psf)\nelif not load_psf and save_psf != None and os.path.isfile(save_psf):\n    raise IOError("File {} exists".format(save_psf))\nelse:\n    print "Not saving the PSF"\n\nif not psfz_ok:\n    plt.imshow(psf, cmap=\'gray\', interpolation="none")\nelif not psfx_ok:\n    psf_xutil = psf.mean(1)\n    plt.plot(psf_xutil)\nelse:\n    l=np.zeros((eye.shape[2], 2*psf.shape[1]))\n    l[:psf.shape[0],:psf.shape[1]]=psf\n    ll=[]\n    for j in range(step*int(eye.shape[2]/psf.shape[0])+1):\n        for k in range(2*psf.shape[0]):\n            ll.append(np.roll(np.roll(l, psf.shape[0]*psf.shape[1]*j/step, axis=0), k, axis=1)[:,int(l.shape[1]/2):int(l.shape[1]*3./2)])\n    \n    D1 = np.hstack(ll) ## This is the dictionary.\n    D2 = np.hstack([i.T.reshape((-1,1)) for i in ll])\n    \n###\n### ======================== Generate stuff\n###\n\n## Generate measurement matrix\n# Generate basis\nif n_mes == 0:\n    b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)\nelse:\n    print "Truncating"\n    b=cstools.generate_fourier_basis(101, n_mes+1, sample=False, oldmethod=True)\n    fou = fou[:,:,:n_mes]\n\nB = cstools.build_2dmodel(b, psf.mean(0)).dot(D2)\n\n## Apply the 2D model\ndef reco_image(ima, verbose=True, tv=False):\n    rr1 = np.arange(0, eye[0, :,:].shape[0]-psf.shape[1], 1)\n    rr2 = rr1 + psf.shape[1]\n    im = np.zeros_like(eye[0,:,:].T)\n    for (i,j) in enumerate(zip(rr1,rr2)):\n        if verbose and i%100 == 0:\n            print "{}/{}".format(i, len(rr1))\n        (sta,sto) = j\n        #im_sub = fou[ima, range(sta,sto),:].flatten()\n        im_sub = ima[range(sta,sto),:].flatten()\n        if tv:\n            r_sim = spiral_tv(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))\n        else:\n            r_sim = spiral2(im_sub, B).dot(D2.T).reshape((psf.shape[1], -1))\n        #im.append(r_sim)\n        im[:,sta:sto]+=r_sim.T\n    #im = np.vstack(im).T\n    #im = np.zeros(())\n    return im\n\n###\n### =========================== Reconstruction step\n###\nprint "Reconstructing"\nim = reco_image(fou[frame_to_reconstruct,:,:], verbose=False) # Launch a full stack reconstruction\n\n###\n### =========================== Saving step\n###\nnp.savetxt(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct)), im)')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/07_reconstruction_3d/recombine.py', '## Maxime W., Jul 2016, GPLv3+\n## This is actually the worker to combine the movie to a 3D stack\n\n# ==== Imports\nimport sys, os\nimport numpy as np\nfrom libtiff import TIFF\n\ndim1 = 512\ndim2 = 101\nscl = 500 # Multiply by this factor before saving.\n\n\nprint "Saving a stack of {} elements, {} px".format(dim2, dim1)\n\n# ==== Variables\n(none, frame_bn, out_dir) = sys.argv\ntmp_dir = os.path.join(out_dir, \'\')\nout_file = os.path.join(out_dir, "{}.tif".format(frame_bn))\nremove_neg = True # if True, negative elements in the reconstruction will be set to zero.\n\n# ==== Checking inputs\n## Input files\ninp_files = [i for i in os.listdir(tmp_dir) if i.startswith(frame_bn)]\nif inp_files == []:\n    raise IOError(\'No files found in {} with name starting with {}. ABORTING\'.format(tmp_dir, frame_bn))\n    \n## Output file\nif os.path.isfile(out_file):\n    im = TIFF.open(out_file, \'r\')\n    for i in im.iter_images():\n        frame0=i\n        break\n    im.close()\n    if frame0.shape[0] >= len(inp_files):\n        print "A TIFF stack already exists and has at least as many pixels as what we"\n        print "were aiming to reconstruct. EXITING."\n        sys.exit(0)\n\n## ==== Saving image\nprint(\'Image dimensions are hardcoded so far. I know this is stupid.\')\nidx = [int(i.split(\'.\')[-2]) for i in inp_files]\nlf = sorted(zip(inp_files, idx), key=lambda x: x[1])\nim = np.zeros((len(lf), dim1, dim2))\nfor (i,j) in enumerate(lf):\n    f=np.genfromtxt(os.path.join(tmp_dir, j[0]))\n    im[i,:,:]=f.T\nprint "Loaded {} planes".format(len(lf))\n\nif remove_neg:\n    print "Negative elements were set to zero"\n    im[im<0]=0 ## Bruteforce.\ntif = TIFF.open(out_file, \'w\')\ntif.write_image(np.array(np.int_(im*scl).swapaxes(2, 0).swapaxes(1,2), dtype=np.uint16))\ntif.close()')

