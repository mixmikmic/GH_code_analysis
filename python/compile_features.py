import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.decomposition import TruncatedSVD

dnames = ['id','category']
for i in range(1,301):
    dnames.append(str(i))
    
descriptiondf = pd.read_csv('data/text_vecs.csv',names=dnames)

# This takes a little while 
npylines = pd.read_csv('data/allnpy_22.txt', header=None)
npynames = npylines.columns.tolist()
npynames[0] = 'id'
npylines.columns = npynames

# Just need id, not full path
npyid = []
for i in range(0, len(npylines)):
    npyid.append(npylines.id[i].split('.')[0].split('/')[-1])
    
npylines['id'] = npyid

# Find union of vecs:
bothdf = pd.merge(descriptiondf, npylines, how='inner', on=['id'])

# Make subset vectors for easy use later
id_vec = bothdf[bothdf.columns[0:2]]
text_vec = bothdf[bothdf.columns[2:302]]
image_vec = bothdf[bothdf.columns[302:4398]]

# Import hand-labeled categories
hand_df = pd.read_csv('data/lookup_table.csv', delimiter=',')

# Get hand-labeled categories for each item
hand = []
for i in range(0, len(id_vec)):
    a = hand_df[hand_df.SS==id_vec.category[i].strip()].HAND
    if (len(a)>0):
        hand.append(a.values[0])
    else:
        hand.append('NA')
        
handdf = pd.DataFrame()
handdf['hand'] = hand

# Get rid of NA rows (items not mapped to items the client uses)
all_withhand = pd.concat((id_vec, handdf, text_vec, image_vec), axis=1)
all_withhand = all_withhand[all_withhand.hand != 'NA']

# Get new subsets
hand_id_vec = all_withhand[all_withhand.columns[0:4]]
hand_text_vec = all_withhand[all_withhand.columns[4:304]]
hand_image_vec = all_withhand[all_withhand.columns[304:4399]]



hand_id_vec.hand.value_counts().plot.bar(use_index=False)

# Downsample observations
handover1k = all_withhand.hand.value_counts() > 1000
handover1k = handover1k[handover1k]
handover1k = handover1k.index

allhand = pd.DataFrame()
for cat in handover1k:
    mycat = all_withhand[all_withhand.hand==cat]
    mycat = mycat.sample(n=1000, random_state=0)
    allhand = allhand.append(mycat)
    
# Generate new subsets
ds_id_vec = allhand[allhand.columns[0:3]]
ds_text_vec = allhand[allhand.columns[3:303]]
ds_image_vec = allhand[allhand.columns[303:]]

len(ds_id_vec)

allhand.hand.value_counts().plot.bar(use_index=False)

# W2V is fine.
# Plot number of zeros for each image feature, there seem to be a lot of zeros in there...
numZero = ((ds_image_vec==0).sum())
numZero = numZero.sort_values()
numZero.plot.bar()

# 4k features with many zeros --> 300 dense features
svd = TruncatedSVD(n_components=300, n_iter=7, random_state=0)
# ds_id_vec = allhand[allhand.columns[0:3]]
# ds_text_vec = allhand[allhand.columns[3:303]]
# ds_image_vec = allhand[allhand.columns[303:]]

allhand_image = svd.fit_transform(ds_image_vec)
allhand_image = pd.DataFrame(allhand_image)
allhand_image.index = ds_image_vec.index

allhand_reduced = pd.concat([ds_id_vec, ds_text_vec, allhand_image], axis=1)
allhand = allhand_reduced

allhand.shape





# This takes a long time. ~30 min for full vecs, but those aren't used currently.
bothdf.to_csv('data/both_vectors.csv')

# Output vectors for just the 11k we'll use. 
allhand_reduced.to_csv('data/handlabeled_vectors_1k.csv')



