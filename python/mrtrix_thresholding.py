import numpy as np
from scipy.stats import poisson
from scipy.ndimage.measurements import label
from scipy.stats import mode
import nibabel as nib
from nipype.interfaces import mrtrix as mrt

theDir = "/Users/srothmei/Desktop/charite/toronto/Adalberto/debug/"
csd_file = theDir + "csd8.nii.gz"
csd_file_random = theDir + "csd_random.nii.gz"

singleVoxelTracks = theDir + 'QL_10011_tracks_singleVox.tck'
singleVoxelTracksRandom = theDir + 'QL_10011_tracks_singleVox_random.tck'

singleVoxelTDI = theDir + 'QL_10011_tracks_singleVox_TDI.nii.gz'
singleVoxelRandomTDI = theDir + 'QL_10011_tracks_singleVox_random_TDI.nii.gz'

tracksPerVoxel = 200*500
trackPerVoxelRandom = tracksPerVoxel * 20
confidenceThreshold = 95 # Value in %

csd8 = nib.load(csd_file)
csd8_data = csd8.get_data()

for i in range(96):
    for j in range(96):
        for k in range(61):
            if np.sum(csd8_data[i,j,k,:] == np.zeros_like(csd8_data[30,40,32,:])) < 45:
                csd8_data[i,j,k,:] = np.zeros_like(csd8_data[30,40,32,:])
                csd8_data[i,j,k,0] = 1.

csd8 = nib.Nifti1Image(csd8_data, csd8.affine, csd8.header)
nib.save(csd8, csd_file_random)

tracker = mrt.StreamlineTrack()
tracker.inputs.inputmodel = 'SD_PROB'
tracker.inputs.stop = True
tracker.inputs.minimum_tract_length = 30
tracker.inputs.no_mask_interpolation = True
tracker.inputs.step_size = 0.2
tracker.inputs.unidirectional = True #Maybe off?
tracker.inputs.seed_file = theDir + 'seedmask10011_1mm_test.nii.gz'
#tracker.inputs.seed_spec = [-50,-6,23,1]
tracker.inputs.include_file = theDir + 'targetmask1001_1mm.nii.gz'
tracker.inputs.mask_file = theDir + 'wmmask_68_1mm.nii.gz'

# Now first for the "informed case"
tracker.inputs.in_file = csd_file
tracker.inputs.out_file = singleVoxelTracks
tracker.inputs.desired_number_of_tracks = tracksPerVoxel
# Perform the fiber tracking
#tracker.run()
tracker.cmdline

# Secondly the "uninformed case"
#tracker.inputs.in_file = csd_file_random
#tracker.inputs.out_file = singleVoxelTracksRandom
#tracker.inputs.desired_number_of_tracks = trackPerVoxelRandom
#tracker.run()

tdi = mrt.Tracks2Prob()
tdi.inputs.fraction = False
tdi.inputs.template_file = theDir + 'seedmask10011_1mm.nii.gz'

tdi.inputs.in_file = singleVoxelTracks
tdi.inputs.out_filename = singleVoxelTDI
tdi.run()

tdi.inputs.in_file = singleVoxelTracksRandom
tdi.inputs.out_filename = singleVoxelRandomTDI
tdi.run()

tdi_informed = nib.load(singleVoxelTDI)
tdi_informed_data = tdi_informed.get_data()

tdi_random = nib.load(singleVoxelRandomTDI)
tdi_random_data = tdi_random.get_data()

# Equation (4)
meanV = (tracksPerVoxel * tdi_random_data) / float(trackPerVoxelRandom)

# Equation (5)
stdV = np.sqrt(meanV)

#Quick and dirty loop
Z_Map = np.zeros_like(meanV, dtype='float64')
raw_P_Map = np.zeros_like(Z_Map)
for x in range(np.shape(meanV)[0]):
    for y in range(np.shape(meanV)[1]):
        for z in range(np.shape(meanV)[2]):
            if stdV[x,y,z] > 0.0:
                # Equation (7)
                tmp = (tdi_informed_data[x,y,z] - meanV[x,y,z]) / stdV[x,y,z]
                #if tmp > 0.0:
                #    Z_Map[x,y,z] = np.log(tmp)
                #else:
                Z_Map[x,y,z] = tmp
                #Omit the negative Z-values
                if tmp >= 0.0:
                    k = tdi_informed_data[x,y,z]
                    mu = meanV[x,y,z]
                    raw_P_Map[x,y,z] = 1 - poisson.pmf(k, mu)

# Thresholded P-map
P_Map_thresholded = np.zeros_like(raw_P_Map, dtype="int16")
P_Map_thresholded[raw_P_Map >= confidenceThreshold/100.0] = 1

                    
# Some debug/visual stuff
zmapImage = nib.Nifti1Image(Z_Map, tdi_informed.affine, tdi_informed.header)
nib.save(zmapImage, theDir + 'Zmap.nii.gz')

pmapImage = nib.Nifti1Image(raw_P_Map, tdi_informed.affine, tdi_informed.header)
nib.save(pmapImage, theDir + 'Pmap_raw.nii.gz')

pmapThres = nib.Nifti1Image(P_Map_thresholded, tdi_informed.affine, tdi_informed.header)
nib.save(pmapThres, theDir + 'Pmap_thresholded.nii.gz')

structure = np.ones((3,3,3))
tmp = np.zeros_like(P_Map_thresholded, dtype="int16")
tmp, bar = label(P_Map_thresholded, structure)
modal_val, modal_count = mode(tmp[tmp>0], axis=None)
P_Map_thresholded_fpCorr = np.zeros_like(P_Map_thresholded)
P_Map_thresholded_fpCorr[tmp == modal_val] = 1

# Debugging / Testing
pmapThresFP = nib.Nifti1Image(P_Map_thresholded_fpCorr, tdi_informed.affine, tdi_informed.header)
nib.save(pmapThresFP, theDir + 'Pmap_thresholded_FPcorr.nii.gz')

# First invert the mask to apply it with MRTrix's tracks_filter
P_Map_thresholded_fpCorr_inv = np.invert(P_Map_thresholded_fpCorr.astype(bool)).astype(int)
# Now save it to use it!
pmapThresFP_inv = nib.Nifti1Image(P_Map_thresholded_fpCorr_inv, tdi_informed.affine, tdi_informed.header)
nib.save(pmapThresFP_inv, theDir + 'Pmap_thresholded_FPcorr_inv.nii.gz')

tracksFilter = mrt.FilterTracks()
tracksFilter.inputs.in_file = singleVoxelTracks
tracksFilter.inputs.no_mask_interpolation = True
tracksFilter.inputs.exclude_file = theDir + 'Pmap_thresholded_FPcorr_inv.nii.gz'
tracksFilter.inputs.invert = False
tracksFilter.inputs.out_file = theDir + 'QL_10011_tracks_singleVox_filt.tck'

tracksFilter.run()

trackVisConv = mrt.MRTrix2TrackVis()
trackVisConv.inputs.in_file = singleVoxelTracks
trackVisConv.inputs.image_file = csd_file
trackVisConv.inputs.out_filename = singleVoxelTracks[:-3] + "trk"
trackVisConv.run()

trackVisConv.inputs.in_file = theDir + 'QL_10011_tracks_singleVox_filt.tck'
trackVisConv.inputs.out_filename = theDir + 'QL_10011_tracks_singleVox_filt.trk'
trackVisConv.run()

excluded_players = "1;2;5"


squery = "SELECT id, name FROM players "

if excluded_players is not None:
    excluded_players = excluded_players.split(';')
    squery += "WHERE "

    for pID in excluded_players:
        squery += "ID != " + pID + " AND "
    # Remove the last 'AND'
    squery = squery[:-5]
    
print squery

tck_filesnames = list()
tck_filesnames.append('LM.tck')
tck_filesnames.append('Garius.tck')

for fname in tck_filesnames:
    print fname

print tck_filesnames[-1]



