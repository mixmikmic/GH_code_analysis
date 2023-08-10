get_ipython().magic('matplotlib inline')
# load stuff
import matplotlib.pyplot as plt
import sys, time, os
import spams
import numpy as np
from libtiff import TIFF, TIFFfile
from sklearn.feature_extraction.image import extract_patches
sys.path.append("../../3_code/")
import cstools
reload(cstools)

n_mes = 0
train_size = 128
patch_width = 20
compute_dict = False ## If True, the dictionary is learned again.

# Define paths
bn = "../../4_data/lattice-lightsheet/"
p_psf = os.path.join(bn, "161013_cstest/eyefullfov20um.tif")
p_eye = os.path.join(bn, "10182016_phal_cs/cell1/eyemat_200ms_20um_CamB_ch0_stack0000_1nm_0000000msec_0004907557msecAbs.tif")
p_fou = os.path.join(bn, "10182016_phal_cs/cell1/f_1stlineoff_200ms_CamB_ch0_stack0000_1nm_0000000msec_0005398464msecAbs.tif")
p_dic = "../../5_outputs/20161209_lattice_dict2D/161209_2Ddict_3000.npy"

# useful function
def spiral(m, b):
    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000, noisetype='poisson')
def spiral2(m, b):
    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000)
def spiral_tv(m, b):
    return cstools.reconstruct_1Dspiral(m, b, maxiter=1000, penalty='tv')

# load data
eye = cstools.read_tif(p_eye, div2=True)
fou_f = cstools.read_tif(p_fou, div2=True, offset2=True)
psf_f = cstools.read_tif(p_psf, div2=True)

# Generate basis
if n_mes == 0:
    b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)
else:
    b=cstools.generate_fourier_basis(101, n_mes+1, sample=False, oldmethod=True)
    fou = fou[:,:,:n_mes]
    
# Training/testing set
tra = eye[train_size:,:,:]
fou = fou_f[:train_size,:,:]

patch_size = (1,patch_width,tra.shape[2])
patches=extract_patches(tra, patch_size)
magnitude = []
patches_flat = np.zeros((patches.shape[0]*patches.shape[1], patches.shape[4], patches.shape[5]))
patches_flatflat = np.zeros((patches.shape[0]*patches.shape[1], patches.shape[4]*patches.shape[5]))
k = 0
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        p = patches[i,j,0,0,:,:]
        magnitude.append((p**2).sum())
        patches_flat[k,:,:]=p
        patches_flatflat[k,:] = p.flatten()
        k += 1
        
magnitude = np.array(magnitude)
patches_high = patches_flat[magnitude>np.percentile(magnitude, 60),:,:] ## Keep top 15%
patches_highflat = patches_flatflat[magnitude>np.percentile(magnitude, 60),:] ## Keep top 15%

plt.figure(figsize=(18,6))
for (i,p) in enumerate(patches_high[:45]):
    plt.subplot(3,15,i+1)
    plt.imshow(patches_high[75*i].T, cmap='gray', interpolation='none')
print "A few patches (before flattening)"

# Learn a dictionary on that. (Keep 20%)
if compute_dict:
    paramTrain = {'K' : 6000, 'numThreads' : 30, 'iter' : 30}
    X=np.asfortranarray(np.float_(patches_highflat).T)

    print "Starting learning procedure, please be patient."
    tic = time.time()
    D = spams.nmf(X,**paramTrain)
    toc = time.time()
 
    print "Dictionary learning performed in {} s".format(toc-tic) # Took 7 hrs for a dict. with 3000 atoms (and 4551 pixels/atom)
    np.save(p_dic, D)
else:
    print "Loading the dictionary, no need to recompute it"
    D = np.load(p_dic)

# Show the patches after learning
plt.figure(figsize=(18,6))

for i in range(D.shape[0])[:45]:
    plt.subplot(3,15,i+1)
    plt.imshow(D[:,i].reshape(patch_width, -1).T, cmap='gray', interpolation='none')
print "A few learned patches (before flattening). There are {} patches in the dictionary".format(D.shape[1])

B = cstools.build_2dmodel(b, np.ones(patch_width)).dot(D)

plt.imshow(B, cmap='gray', interpolation='none')

# ==== 3. Subdivide the simulated image & reconstruct (with sliding windows)
def reco_image2(ima, verbose=True, solver=spiral):
    im = np.zeros((ima.shape[0], b.shape[1]))
    vec_counts = np.zeros(ima.shape[0])
    for i in range(eye[0, :,:].shape[0]):
        if verbose and i%100 == 0:
            print "-- {}/{}".format(i, eye[0, :,:].shape[0])
        (sta,sto) = (i-patch_width, i)
        im_sub = ima[range(sta,sto),:].flatten()
        #r_sim = spiral2(im_sub, B).reshape((len(psf_xutil),-1))
        r_sim = solver(im_sub, B).dot(D.T).reshape((patch_width, -1))
        im[range(sta,sto),:]+=r_sim
        vec_counts[range(sta,sto)]+=1
        
    for (i,v) in enumerate(vec_counts): # Average each row
        im[i,:] /= v
    return im

im = reco_image2(fou_f[:,78,:], verbose=True)

im_poi = reco_image2(fou_f[:,78,:], verbose=True, solver=spiral2)
im_tv  = reco_image2(fou_f[:,78,:], verbose=True, solver=spiral_tv)

plt.figure(figsize=(18,10))
plt.subplot(221)
plt.imshow(im.T[::-1,:], cmap='gray')
plt.subplot(222)
plt.imshow(im_poi.T[::-1,:], cmap='gray')
plt.subplot(223)
plt.imshow(im_tv.T[::-1,:], cmap='gray')
plt.subplot(224)
plt.imshow(eye[:,78,:].T, cmap='gray')
plt.title("Reference")

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
    D2_psf = np.hstack([i.T.reshape((-1,1)) for i in ll])

B = cstools.build_2dmodel(b, psf.mean(0)).dot(D2_psf)
patch_width = psf.shape[1]

plt.imshow(D2_psf, cmap='gray', interpolation='none')

# ==== 3. Subdivide the simulated image & reconstruct (with sliding windows)
def reco_image_psf(ima, verbose=True, solver=spiral):
    im = np.zeros((ima.shape[0], b.shape[1]))
    vec_counts = np.zeros(ima.shape[0])
    for i in range(eye[0, :,:].shape[0]):
        if verbose and i%100 == 0:
            print "-- {}/{}".format(i, eye[0, :,:].shape[0])
        (sta,sto) = (i-patch_width, i)
        im_sub = ima[range(sta,sto),:].flatten()
        #r_sim = spiral2(im_sub, B).reshape((len(psf_xutil),-1))
        r_sim = solver(im_sub, B).dot(D2_psf.T).reshape((patch_width, -1))
        im[range(sta,sto),:]+=r_sim
        vec_counts[range(sta,sto)]+=1
        
    for (i,v) in enumerate(vec_counts): # Average each row
        im[i,:] /= v
    return im

#im = reco_image_psf(fou_f[:,78,:], verbose=True)
im_poi = reco_image_psf(fou_f[:,78,:], verbose=True, solver=spiral2)
im_tv = reco_image_psf(fou_f[:,78,:], verbose=True, solver=spiral_tv)
## Also try with other solvers

plt.figure(figsize=(18,16))
plt.subplot(421);plt.imshow(eye[:,78,:].T, cmap='gray');plt.title("Reference")
plt.subplot(422);plt.imshow(im.T[::-1,:], cmap='gray', interpolation='none');plt.title("Gaussian likelihood")
plt.subplot(423);plt.imshow(im_poi.T[::-1,:], cmap='gray', interpolation='none');plt.title("Poisson likelihood")
plt.subplot(424);plt.imshow(im_tv.T[::-1,:], cmap='gray', interpolation='none');plt.title("Total variation")

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/09_lattice_phalloidin/reconstruction5.sh', '#!/bin/sh\n## /!\\ This code is generated by a Jupyter script, do not edit it directly.\n## It is designed to run reconstructions on a very specific dataset.\n## It should be adapted carefully before playing on the input stuff\n\necho "==== DISCLAIMER ===="\necho "Have you installed the following packages?: virtualenv, numpy, scipy libtiff, joblib, pycsalgos, h5py, pySPIRALTAP"\necho "Have you run the following commands to load the packages?"\necho "$ module load Python/2.7.11"\necho "$ source ~/.local/bin/virtualenvwrapper.sh"\necho "$ export WORKON_HOME=~/.envs"\necho "$ workon dict-optim"\n\nN_FRAMES_IN_STACK=256\nemail="maxime.woringer@pasteur.fr"\nmaxparalleljobs=100\n\necho \'Running on tars\'\nsbatch --mail-type=BEGIN,END --mail-user=$email --array=0-$N_FRAMES_IN_STACK%$maxparalleljobs ../8_cluster/tars/09_lattice_phalloidin/reconstruction5_init.sh')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/09_lattice_phalloidin/reconstruction5_init.sh', '#!/bin/sh\n#SBATCH --qos=fast \n#SBATCH -N 1\n#SBATCH -c 12\n#SBATCH -p common,dedicated\n#SBATCH -o ../8_cluster/tars/09_lattice_phalloidin/log_reconstruction.log -e ../8_cluster/tars/09_lattice_phalloidin/log_reconstruction.err\n# By Maxime W., Nov. 2016, GPLv3+\n# /!\\ DO NOT EDIT THIS FILE. IT HAS BEEN GENERATED BY A SCRIPT\n# Script is ../2_simulations/51. ...ipynb\n\n## This script to be called by SBATCH, do not call it directly, it will not work.\nsrun ~/.envs/dict-optim/bin/python ../8_cluster/tars/09_lattice_phalloidin/reconstruction5.py ${SLURM_ARRAY_TASK_ID}')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/09_lattice_phalloidin/reconstruction5.py', '\nimport sys, os\nimport numpy as np\nfrom libtiff import TIFF\nsys.path.append("../3_code/")\nimport cstools\n#reload(cstools)\n\nn_mes = 10 ## Compression factor (a number of frames)\nsave_prefix = "phalloidin{}".format(n_mes)\n\n\n## tars-specific parameters\nframe_to_reconstruct = int(sys.argv[1])\n\n\n# useful function\ndef spiral(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000, noisetype=\'poisson\')\ndef spiral2(m, b):\n    return cstools.reconstruct_1Dspiral(m, b, maxiter=2000)\n\n# Define paths\nbn = "../4_data/lattice-lightsheet/"\np_psf = os.path.join(bn, "161013_cstest/eyefullfov20um.tif")\np_eye = os.path.join(bn, "10182016_phal_cs/cell1/eyemat_200ms_20um_CamB_ch0_stack0000_1nm_0000000msec_0004907557msecAbs.tif")\np_fou = os.path.join(bn, "10182016_phal_cs/cell1/f_1stlineoff_200ms_CamB_ch0_stack0000_1nm_0000000msec_0005398464msecAbs.tif")\n#p_dic = "../../5_outputs/20161209_lattice_dict2D/161209_2Ddict_3000.npy"\n\nout_folder = "../5_outputs/12_reconstructions_march_phalloidin"\n\n## Do not reconstruct if we already have it\nif os.path.exists(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct))):\n    print "reconstruction {} exists, aborting".format("{}.{}.txt".format(save_prefix, frame_to_reconstruct))\n    sys.exit(0)\n\n# load data\neye = cstools.read_tif(p_eye, div2=True)\nfou_f = cstools.read_tif(p_fou, div2=True, offset2=True)\npsf_f = cstools.read_tif(p_psf, div2=True)\n\nfou = fou_f\n\n# Generate basis\nif n_mes == 0:\n    b=cstools.generate_fourier_basis(101, 50+1, sample=False, oldmethod=True)\nelse:\n    b=cstools.generate_fourier_basis(101, n_mes+1, sample=False, oldmethod=True)\n    fou = fou[:,:,:n_mes]\n    \n# Training/testing set\n#tra = eye[train_size:,:,:]\n#fou_f[:train_size,:,:] ## We perform a full-size reconstruction\n\n###\n### ===================== Extract a PSF and build measurement matrix\n###\n## 1. Extract a PSF\npsfz_ok = True # Flag to switch between calibration and generation of the PSF\npsfx_ok = True\nsave_psf = "../5_outputs/psf_models/lattice_161114" ## Set to None to avoid saving\nload_psf = True ## Load the PSF from the save_psf file.\n\nstep = 8 ## A fraction of the size of the PSF in z. Represent the shift in the final dictionary.\npsf_x = 205\npsf_y = range(363, 367)\npsf_z = (82, 95)\n\nif load_psf:\n    psf=np.load(save_psf+".npy")\n    print "Loaded PSF from file {}.npy".format(save_psf)\nelse:\n    psf = eye.mean(0)[psf_y,psf_z[0]:psf_z[-1]].T\n    psf -= psf.min()\n\nif save_psf != None and not load_psf and not os.path.isfile(save_psf): ## Save the PSF if needed\n    np.save(save_psf, psf)\n    print "PSF saved on file {}".format(save_psf)\nelif not load_psf and save_psf != None and os.path.isfile(save_psf):\n    raise IOError("File {} exists".format(save_psf))\nelse:\n    print "Not saving the PSF"\n\nif not psfz_ok:\n    plt.imshow(psf, cmap=\'gray\', interpolation="none")\nelif not psfx_ok:\n    psf_xutil = psf.mean(1)\n    plt.plot(psf_xutil)\nelse:\n    l=np.zeros((eye.shape[2], 2*psf.shape[1]))\n    l[:psf.shape[0],:psf.shape[1]]=psf\n    ll=[]\n    for j in range(step*int(eye.shape[2]/psf.shape[0])+1):\n        for k in range(2*psf.shape[0]):\n            ll.append(np.roll(np.roll(l, psf.shape[0]*psf.shape[1]*j/step, axis=0), k, axis=1)[:,int(l.shape[1]/2):int(l.shape[1]*3./2)])\n    \n    D1 = np.hstack(ll) ## This is the dictionary.\n    D2_psf = np.hstack([i.T.reshape((-1,1)) for i in ll])\n\npatch_width = psf.shape[1]\n    \n###\n### ======================== Generate stuff\n###\nB = cstools.build_2dmodel(b, psf.mean(0)).dot(D2_psf)\n\n\n## Apply the 2D model\n# ==== 3. Subdivide the simulated image & reconstruct (with sliding windows)\ndef reco_image_psf(ima, verbose=True, solver=spiral):\n    im = np.zeros((ima.shape[0], b.shape[1]))\n    vec_counts = np.zeros(ima.shape[0])\n    for i in range(eye[0, :,:].shape[0]):\n        if verbose and i%100 == 0:\n            print "-- {}/{}".format(i, eye[0, :,:].shape[0])\n        (sta,sto) = (i-patch_width, i)\n        im_sub = ima[range(sta,sto),:].flatten()\n        #r_sim = spiral2(im_sub, B).reshape((len(psf_xutil),-1))\n        r_sim = solver(im_sub, B).dot(D2_psf.T).reshape((patch_width, -1))\n        im[range(sta,sto),:]+=r_sim\n        vec_counts[range(sta,sto)]+=1\n        \n    for (i,v) in enumerate(vec_counts): # Average each row\n        im[i,:] /= v\n    return im\n###\n### =========================== Reconstruction step\n###\nprint "Reconstructing"\nim = reco_image_psf(fou[frame_to_reconstruct,:,:], verbose=False) # Launch a full stack reconstruction\n\n###\n### =========================== Saving step\n###\nnp.savetxt(os.path.join(out_folder, "{}.{}.txt".format(save_prefix, frame_to_reconstruct)), im)')

get_ipython().run_cell_magic('writefile', '../../8_cluster/tars/09_lattice_phalloidin/recombine.py', '## Maxime W., Jul 2016, GPLv3+\n## This is actually the worker to combine the movie to a 3D stack\n\n# ==== Imports\nimport sys, os\nimport numpy as np\nfrom libtiff import TIFF\n\ndim1 = 256\ndim2 = 101\nscl = 1000 # Multiply by this factor before saving.\n\n\nprint "Saving a stack of {} elements, {} px".format(dim2, dim1)\n\n# ==== Variables\n(none, frame_bn, out_dir) = sys.argv\ntmp_dir = os.path.join(out_dir, \'\')\nout_file = os.path.join(out_dir, "{}.tif".format(frame_bn))\nremove_neg = True # if True, negative elements in the reconstruction will be set to zero.\n\n# ==== Checking inputs\n## Input files\ninp_files = [i for i in os.listdir(tmp_dir) if i.startswith(frame_bn)]\nif inp_files == []:\n    raise IOError(\'No files found in {} with name starting with {}. ABORTING\'.format(tmp_dir, frame_bn))\n    \n## Output file\nif os.path.isfile(out_file):\n    im = TIFF.open(out_file, \'r\')\n    for i in im.iter_images():\n        frame0=i\n        break\n    im.close()\n    if frame0.shape[0] >= len(inp_files):\n        print "A TIFF stack already exists and has at least as many pixels as what we"\n        print "were aiming to reconstruct. EXITING."\n        sys.exit(0)\n\n## ==== Saving image\nprint(\'Image dimensions are hardcoded so far. I know this is stupid.\')\nidx = [int(i.split(\'.\')[-2]) for i in inp_files]\nlf = sorted(zip(inp_files, idx), key=lambda x: x[1])\nim = np.zeros((len(lf), dim1, dim2))\nfor (i,j) in enumerate(lf):\n    f=np.genfromtxt(os.path.join(tmp_dir, j[0]))\n    im[i,:,:]=f.T\nprint "Loaded {} planes".format(len(lf))\n\nif remove_neg:\n    print "Negative elements were set to zero"\n    im[im<0]=0 ## Bruteforce.\ntif = TIFF.open(out_file, \'w\')\ntif.write_image(np.array(np.int_(im*scl).swapaxes(2, 0).swapaxes(1,2), dtype=np.uint16))\ntif.close()')

