import numpy as np
import pandas as pd
import os
import collections
import pickle

df = pd.read_csv('../data/CSVFILES/annotations_enhanced.csv')
df['ix'] = range(df.shape[0])

patients = collections.OrderedDict()
dfsub_shape = []

for directory in [d for d in os.listdir('../data/') if 'subset' in d]:
    patients_by_dir = [f.replace('.mhd','') for f in os.listdir('../data/'+directory) if '.mhd' in f]
    patients[directory] = collections.OrderedDict()
    for patient in patients_by_dir:
        patients[directory][patient] = collections.OrderedDict()
        dfsub = df[df['seriesuid']==patient]
        dfsub_shape.append(dfsub.shape[0])
        count = max(6,6*dfsub.shape[0]*2)
        if count==6:
            patients[directory][patient]['random'] = count
        else:
            patients[directory][patient]['random'] = count/2
            patients[directory][patient]['true'] = count/2

with open('./voxel_to_patient_dict.pickle', 'wb') as handle:
    pickle.dump(patients, handle, protocol=pickle.HIGHEST_PROTOCOL)
print ('Dictionary SAVED')

patients_subset2 = [f.replace('.mhd','') for f in os.listdir('../data/subset2/') if '.mhd' in f]

print ('Patient',list(patients[list(patients.keys())[0]].keys())[0])
list(patients[list(patients.keys())[0]].keys())[0] == patients_subset2[0]

subset2true = np.load('../data/LUNA_voxels/subset2Xtrue.npy')
subset2random = np.load('../data/LUNA_voxels/subset2Xrandom.npy')

print (subset2true.shape)
print (subset2random.shape)

allvoxelscount = []
for patient in patients['subset2'].keys():
    allvoxelscount.extend(patients['subset2'][patient].values())
sum(allvoxelscount)

subset2random.shape[0]+subset2true.shape[0]

print ('Subsets in dictionary..')
list(patients.keys())

print ('Patients in subset..')
list(patients[list(patients.keys())[0]].keys())[0:3]

print ('Categories in patient..')
list(patients[list(patients.keys())[0]][list(patients[list(patients.keys())[0]].keys())[1]].keys())

num_voxels = []
for subset in patients:
    for patient in patients[subset]:
        num_voxels.append(sum(patients[subset][patient].values()))
print (len(num_voxels),max(num_voxels),min(num_voxels))
import matplotlib.pyplot as plt
plt.hist(num_voxels)
plt.show()



