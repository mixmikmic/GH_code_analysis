import subprocess
import os
import numpy as np

#Your working directory containing the cloned code
parentDir = 'C:/Users/username/Documents/ZebrafishBrainRegistration_Zbrain_ANTs/'

#Path to the ANTs exe-files (Windows)
binPath = parentDir + 'ANTs_2.1.0_Windows/'

#Path to down-sampled Z-brain stacks (5 um isotropic resolution)
pathZbrain_H2B = parentDir + 'ZBrains/zbrain-H2B-RFP_6dpf_mean10fish_8bit_xyz5um.nrrd'
pathZbrain_Cyto = parentDir + 'ZBrains/zbrain-tERK-8bit_xyz5um.nrrd'

# nMLF ablations, Cyto fish,
ablationName = 'nMLF'
h2b = False #set True if aligning (nuclear) H2B-expressing markers, False is cytosolically expressed marker

rawDirList =  ['2016-10-18fish1/pre/',               '2016-10-18fish1/post/',               
               '2016-10-18fish3/pre/',\
               '2016-10-18fish3/post/',\
               
               '2016-10-24fish1/pre/',\
               '2016-10-24fish1/post/'           
              ]
signalType = 'Fstim' 
procDirs = []
for dirName in rawDirList:
    procDirs.append(parentDir + 'raw_brains/' + dirName)        

for i_exp in range(1,len(procDirs),2):
    procDir = procDirs[i_exp]
    print str(1+ (i_exp-1)/2) + ' of ' + str(len(procDirs)/2)  
    os.chdir(procDir)
    title = os.path.basename(os.path.split(os.path.split(os.path.split(procDir)[0])[0])[0])

    filename_fixed = procDirs[i_exp - 1] + 'ref_xyz5um.nrrd' #pre-ablation file, fixed (target)
    filename_movingNRRD = procDir + 'ref_xyz5um.nrrd' #post-ablation (moving)

    filename_aligned = procDir +'refpost2pre_xyz5um.nrrd'

    args = binPath + 'antsRegistration.exe '  +             ' --float 0 ' +             ' --dimensionality 3 ' +             ' --interpolation BSpline ' +             ' --use-histogram-matching 1 ' +            ' --winsorize-image-intensities [0.005,0.995] ' +            ' --output [refpost2pre_xyz5um_,' + filename_aligned + '] ' +             ' --initial-moving-transform [' + filename_fixed + ',' + filename_movingNRRD + ',1] ' +            ' --transform Rigid[0.1]' +            ' --metric MI[' + filename_fixed + ',' + filename_movingNRRD + ',1,32,Regular,1.0]' +            ' --convergence [500x250x100,1e-6,10]' +            ' --shrink-factors 4x2x1' +            ' --smoothing-sigmas 2x1x0vox' +            ' --transform Affine[0.1] ' +            ' --metric MI[' + filename_fixed + ',' + filename_movingNRRD + ',1,32,Regular,1.0] ' +            ' --convergence [500x250x100,1e-6,10] ' +            ' --shrink-factors 4x2x1 ' +            ' --smoothing-sigmas 2x1x0vox' +            ' --transform SyN[0.1,3,0]' +            ' --metric CC[' + filename_fixed + ',' + filename_movingNRRD + ',1,4] ' +            ' --convergence [200x100x50,1e-6,10]' +            ' --shrink-factors 3x2x1' +            ' --smoothing-sigmas 2x1x0vox'
    logStr = subprocess.check_output(args) 

for i_exp in range(1,len(procDirs),2):
    procDir = procDirs[i_exp]
    print str(1+ (i_exp-1)/2) + ' of ' + str(len(procDirs)/2)    
    os.chdir(procDir)
    
    title = os.path.basename(os.path.split(os.path.split(procDir)[0])[0])

    # pre- to -post
    transform1 = procDir + 'refpost2pre_xyz5um_1Warp.nii.gz'
    transform0 = procDir + 'refpost2pre_xyz5um_0GenericAffine.mat'

    # Sub-sampled dFF activity file warping
    filename_fixed = procDirs[i_exp - 1] + 'ref_xyz5um.nrrd' #pre-ablation file, fixed (target)
    filename_moving = procDir  + title + '_meanActivity' + signalType + '_xyz5um.nrrd'
    filename_warped = procDir + title + '_meanActivity' + signalType + '_xyz5um_warp2pre.nrrd'
    
    argsTransform = binPath + 'antsApplyTransforms.exe' +                     ' --dimensionality 3 ' +                     ' --input ' + filename_moving +             ' --reference-image ' + filename_fixed +             ' --output ' + filename_warped +             ' --interpolation Linear ' +             ' --transform ' + transform1 +             ' --transform ' + transform0 +             ' --default-value 0'
    logStrTransform = subprocess.check_output(argsTransform)

for i_exp in range(1,len(procDirs),2):
    procDir = procDirs[i_exp]
    title = os.path.basename(os.path.split(os.path.split(procDir)[0])[0])
    
    stack_pre, options = nrrd.read(procDirs[i_exp - 1] + title + '_meanActivity'+signalType+'_xyz5um.nrrd')
    stack_post, options = nrrd.read(procDir + title + '_meanActivity' + signalType + '_xyz5um_warp2pre.nrrd')
    
    stackDiff = stack_pre.astype('float32') - stack_post.astype('float32') 
    
    nrrd.write(procDir  + title + '_'+signalType+'_pre-post_warp_xyz5um.nrrd',stackDiff,options)

get_ipython().run_cell_magic('time', '', "if h2b:\n    filename_fixed = pathZbrain_H2B\nelse: \n    filename_fixed = pathZbrain_Cyto\n    \nfor i_exp in range(1,len(procDirs),2):\n    procDir = procDirs[i_exp - 1]\n    print str(1+ (i_exp-1)/2) + ' of ' + str(len(procDirs)/2)     \n    os.chdir(procDir)\n\n    filename_movingNRRD = procDir + 'ref_xyz5um.nrrd' #ref file in pre-ablation\n\n    filename_aligned = procDir + 'ref_2ZBrainDownsampled.nrrd'\n\n    args = binPath + 'antsRegistration.exe '  + \\\n            ' --float 0 ' + \\\n            ' --dimensionality 3 ' + \\\n            ' --interpolation BSpline ' + \\\n            ' --use-histogram-matching 1 ' +\\\n            ' --winsorize-image-intensities [0.005,0.995] ' +\\\n            ' --output [ref_pre_2ZBrainDownsampled_,' + filename_aligned + '] ' + \\\n            ' --initial-moving-transform [' + filename_fixed + ',' + filename_movingNRRD + ',1] ' +\\\n            ' --transform Rigid[0.1]' +\\\n            ' --metric MI[' + filename_fixed + ',' + filename_movingNRRD + ',1,32,Regular,1.0]' +\\\n            ' --convergence [500x250x100,1e-6,10]' +\\\n            ' --shrink-factors 4x2x1' +\\\n            ' --smoothing-sigmas 2x1x0vox' +\\\n            ' --transform Affine[0.1] ' +\\\n            ' --metric MI[' + filename_fixed + ',' + filename_movingNRRD + ',1,32,Regular,1.0] ' +\\\n            ' --convergence [500x250x100,1e-6,10] ' +\\\n            ' --shrink-factors 4x2x1 ' +\\\n            ' --smoothing-sigmas 2x1x0vox' +\\\n            ' --transform SyN[0.1,3,0]' +\\\n            ' --metric MI[' + filename_fixed + ',' + filename_movingNRRD + ',1,32,Regular,1.0] ' +\\\n            ' --convergence [200x100x50,1e-6,10]' +\\\n            ' --shrink-factors 4x2x1' +\\\n            ' --smoothing-sigmas 2x1x0vox'\n    logStr = subprocess.check_output(args)    ")

if h2b:
    filename_fixed = pathZbrain_H2B
else: 
    filename_fixed = pathZbrain_Cyto
    
for i_exp in range(1,len(procDirs),2):
    procDir = procDirs[i_exp]
    print str(1+ (i_exp-1)/2) + ' of ' + str(len(procDirs)/2)    
    os.chdir(procDir)
    
    title = os.path.basename(os.path.split(os.path.split(procDir)[0])[0])

    # pre- to Zbrain
    transform1 = procDirs[i_exp - 1] + 'ref_pre_2ZBrainDownsampled_1Warp.nii.gz'
    transform0 = procDirs[i_exp - 1] + 'ref_pre_2ZBrainDownsampled_0GenericAffine.mat'

    filename_moving = procDir  + title + '_'+signalType+'_pre-post_warp_xyz5um.nrrd'
    filename_warped = procDir + title + '_'+signalType+'_pre-post_xyz5um_warp2Zbrain.nrrd'

    argsTransform = binPath + 'antsApplyTransforms.exe' +                     ' --dimensionality 3 ' +                     ' --input ' + filename_moving +             ' --reference-image ' + filename_fixed +             ' --output ' + filename_warped +             ' --interpolation Linear ' +             ' --transform ' + transform1 +             ' --transform ' + transform0 +             ' --default-value 0'
    logStrTransform = subprocess.check_output(argsTransform)



