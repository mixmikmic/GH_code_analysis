import numpy as np
import pandas as pd
import os

PATH_RAW_ARRAY = "../../data/stage1_arrays/"
PATH_VOXELS = "../../data/stage1_TOP_voxels/"

patients = [f[4:] for f in os.listdir(PATH_VOXELS) if 'vox_' in f ]
print ("patient numbers: ", len(patients) )    

from sklearn.cluster import DBSCAN
max_ct_count = []
all_features = []
for num, patient in enumerate(patients):
    patient_vox = np.load(os.path.join(PATH_VOXELS, 'vox_' + patient))      #voxels[filter]
    patient_locs = np.load(os.path.join(PATH_VOXELS, 'cents_' + patient))   #locations[filter]
    patient_sizes = np.load(os.path.join(PATH_VOXELS, 'shapes_' + patient)) #sizes[filter]
    patient_nodule_preds = np.load(os.path.join(PATH_VOXELS, 'preds_' + patient))
    max_ct_count.append(len(patient_nodule_preds[0,:]))
    
    xmax = np.max(patient_nodule_preds[0], axis=0)
    xsd = np.std(patient_nodule_preds[0], axis=0)
#     print ("max malignancy:", xmax)
#     print ("malignancy std:", xsd)
    
    normalized_locs = patient_locs.astype('float32') / patient_sizes.astype('float32')
#     print (normalized_locs)
    loc_from_malig = normalized_locs[ np.argmax(patient_nodule_preds[0],axis=0 )]
#     print ("normalized location from malig:", loc_from_malig)
    
    dist_mat = np.zeros((patient_locs.shape[0], patient_locs.shape[0]))
    for i,loc_a in enumerate(patient_locs):
        for j,loc_b in enumerate(patient_locs):
            dist_mat[i,j] = np.mean(np.abs(loc_a - loc_b))
#     print (dist_mat)

    dbs = DBSCAN(eps=60, min_samples=2, metric='precomputed', leaf_size=2).fit(dist_mat)
    num_clusters = np.max(dbs.labels_) + 1
    num_noise = (dbs.labels_ == -1).sum()
#     print (" num_clusters", num_clusters, "\tnum_noise:", num_noise)
    
    #new feature: sum of malig_scores but normalizing by cluster.
    cluster_avgs = []
    for clusternum in range(num_clusters):
        cluster_avgs.append( patient_nodule_preds[0][dbs.labels_ == clusternum].mean())

    #now get the -1's
    for i,(clusterix,malig) in enumerate(zip(dbs.labels_,patient_nodule_preds[0])):
        if clusterix == -1:
            cluster_avgs.append(malig)

    weighted_sum_malig = np.sum(cluster_avgs)
    weighted_mean_malig = np.mean(cluster_avgs)

    #size of biggest cluster
    sizes = np.bincount(dbs.labels_[dbs.labels_ > 0])
    if len(sizes) > 0:
        maxsize = np.max(sizes)
    else:
        maxsize = 1
    n_nodules = float(patient_locs.shape[0])
    feats = (np.concatenate([(xmax, xsd),loc_from_malig,                            normalized_locs.std(axis=0),                            [float(num_clusters) / n_nodules,                             float(num_noise) / n_nodules,                             weighted_mean_malig,                             float(maxsize) / n_nodules]]))
    all_features.append(feats)
    X = np.stack(all_features)
    if num%200==0:
        print ("Patient:", patient[:-4])
#         print (feats)

col=['max_malig', 'malig_xstd', 'locs_x', 'locs_y','locs_z', 
     'norm_locs_x','norm_locs_y','norm_locs_z','norm_#cluster',
     'norm_#noise','weighted_mean_malig', 'norm_maxsize' ]

df = pd.DataFrame(data=X,index=patients, columns=col)
df.to_csv('./model_v24_feature_matrix.csv')

# print (max_ct_count)

df_feature = pd.read_csv('model_v24_feature_matrix.csv')
df_feature['id'] = df_feature['Unnamed: 0'].apply(lambda x: x.split('.')[0])
print (df_feature.shape)
df_feature.head()



