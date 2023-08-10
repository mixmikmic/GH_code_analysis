# Initial steps:
# Get list of dictionaries
# List index: ICP index
# List entry: Dictionary of ICP information
    # Industry
    # Num. text, image, video
    # Bytes text, image, video

# Only keep most recent data
    # 1. Sort dataframe by timestamp
    # 2. Iterate rows --> Keep row if current entry = empty

# How to compare timestamps
print "string" > "str"
print "20150507" > "20170629"

import pickle

interactions_pickle = '20170629-interactions-mappings.pkl'
with open(interactions_pickle, 'rb') as output:
    (interactions, iidx_to_cdn, cdn_to_iidx, uidx_to_icp, icp_to_uidx) = pickle.load(output)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import csv

# Read in icpclassify file
icpclassify_filepath = 'icpstatistic/icpclassify.txt'
icpclassify_header = ['industry', 'icp']
icpclassify_datatypes = {
    'industry': str,
    'icp': str
}

icpclassify_df = pd.read_csv(icpclassify_filepath, 
                              sep=',', header=None, 
                              names=icpclassify_header,
                              dtype=icpclassify_datatypes)

icpclassify_df.head()

print icpclassify_df.industry.unique()
print icpclassify_df.duplicated().unique()
print icpclassify_df.icp.unique().shape[0]

num_icps = len(icp_to_uidx.keys())
icp_list = icp_to_uidx.keys()

[x for x in icp_to_uidx.values() if icp_to_uidx.values().count(x) >= 2]
# No duplicate mappings. Just to check.

# Next step: try something like this instead
icpclassify_dict = icpclassify_df.set_index('icp').T.to_dict()

print len(icpclassify_dict)

icp_feature_dict_prelim = {icp:features for icp,features                     in icpclassify_dict.iteritems() if icp in icp_to_uidx}
print len(icp_feature_dict_prelim)
print icp_feature_dict_prelim["www.qq.com"] # Testing

# Check all ICPs in dict belong in RecSys
for icp, features in icp_feature_dict_prelim.iteritems():
    assert(icp in icp_to_uidx)

# Check all ICPs in RecSys matrix represented in feature dict
for icp in icp_to_uidx.keys():
    assert (icp in icp_feature_dict_prelim)

# For quick testing
def get_industry(icp):
    if icp in icp_list:
        return icp_feature_dict_prelim[icp]['industry']
    else:
        return 'ICP not in RecSys list'

get_industry('www.17ok.com')

# TODO: Add num page elements, bytes data

# To sort this list based on mapping indices: 
# https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-values-of-the-dictionary-in-python

# Read in icpstatistic file
icpstatistic_filepath = 'icpstatistic/all_icp_statistics.txt'
icpstatistic_header = ['icp', 'textnum', 'imagenum', 'videonum', 'unknownnum',                       'textbytes', 'imagebytes', 'videobytes', 'unknownbytes',                       'createtime', 'ts']
icpstatistic_dtypes = {
    'icp': str,
    'textnum': np.int64,
    'imagenum': np.int64,
    'videonum': np.int64,
    'unknownnum': np.int64,
    'textbytes': np.int64,
    'imagebytes': np.int64,
    'videobytes': np.int64,
    'unknownbytes': np.int64,
    'createtime': str,
    'ts': str
}

icpstatistic_df = pd.read_csv(icpstatistic_filepath, 
                              sep=',', header=None, 
                              names=icpstatistic_header,
                              dtype=icpstatistic_dtypes)

icpstatistic_df.head()

# Sort: recent first --> only keep most recent data
icpstatistic_sorted = icpstatistic_df.sort_values(['ts', 'createtime'], ascending=[False, False])
icpstatistic_sorted.head()

icpstatistic_sorted.describe()

# Web data features: Bin by quartiles? 
# So: 0, 1st quartile (>0), 2nd quartile, 3rd, 4th

# Define quantiles for data feature bins
text_q1 = 0
text_q2 = icpstatistic_sorted['textbytes'].quantile(.25)
text_q3 = icpstatistic_sorted['textbytes'].quantile(.50)
text_q4 = icpstatistic_sorted['textbytes'].quantile(.75)

image_q1 = 0
image_q2 = icpstatistic_sorted['imagebytes'].quantile(.25)
image_q3 = icpstatistic_sorted['imagebytes'].quantile(.50)
image_q4 = icpstatistic_sorted['imagebytes'].quantile(.75)

video_q1 = 0
video_q2 = icpstatistic_sorted['videobytes'].quantile(.70)
video_q3 = icpstatistic_sorted['videobytes'].quantile(.80)
video_q4 = icpstatistic_sorted['videobytes'].quantile(.90)

icpstatistic_sorted['videobytes'].quantile(.54)

# Quantile-bin function:
def text_bin(bytes):
    if bytes <= text_q1: return 'q0'
    elif bytes > text_q1 and bytes < text_q2: return 'q1'
    elif bytes >= text_q2 and bytes < text_q3: return 'q2'
    elif bytes >= text_q3 and bytes < text_q4: return 'q3'
    elif bytes >= text_q4: return 'q4'
    else: return 'Error'
    
def image_bin(bytes):
    if bytes <= image_q1: return 'q0'
    elif bytes > image_q1 and bytes < image_q2: return 'q1'
    elif bytes >= image_q2 and bytes < image_q3: return 'q2'
    elif bytes >= image_q3 and bytes < image_q4: return 'q3'
    elif bytes >= image_q4: return 'q4'
    else: return 'Error'
    
def video_bin(bytes):
    if bytes <= video_q1: return 'q0'
    elif bytes > video_q1 and bytes < video_q2: return 'q1'
    elif bytes >= video_q2 and bytes < video_q3: return 'q2'
    elif bytes >= video_q3 and bytes < video_q4: return 'q3'
    elif bytes >= video_q4: return 'q4'
    else: return 'Error'

video_bin(21175721.0)

# Add web content statistics to feature dict
for entry in icpstatistic_sorted.itertuples():
    icp = entry[1]
    
    # Hasn't used a CDN
    if icp not in icp_to_uidx: 
        continue
        
    # Has no entries yet
    if 'textnum' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['textnum'] = entry[2]
    if 'imagenum' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['imagenum'] = entry[3]
    if 'videonum' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['videonum'] = entry[4]
    if 'unknownnum' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['unknownnum'] = entry[5]
    if 'textbytes' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['textbytes'] = entry[6]
    if 'imagebytes' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['imagebytes'] = entry[7]
    if 'videobytes' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['videobytes'] = entry[8]
    if 'unknownbytes' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['unknownbytes'] = entry[9]
    if 'ts' not in icp_feature_dict_prelim[icp]:
        icp_feature_dict_prelim[icp]['ts'] = entry[11]

for icp, features in icp_feature_dict_prelim.iteritems():
    if 'industry' not in features: print icp
    if 'textnum' not in features: print icp
    if 'imagenum' not in features: print icp
    if 'videonum' not in features: print icp
    if 'unknownnum' not in features: print icp
    if 'textbytes' not in features: print icp
    if 'imagebytes' not in features: print icp
    if 'videobytes' not in features: print icp
    if 'unknownbytes' not in features: print icp

# Manually adding info for www.chinacourt.org - got listed as "chinacourt.org" in icpstatistic
# Please forgive me for this shitty code
for entry in icpstatistic_sorted.itertuples():
    icp_raw = entry[1]
    
    if 'chinacourt.org' in icp_raw:
        icp = 'www.chinacourt.org'
        
        if 'textnum' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['textnum'] = entry[2]
        if 'imagenum' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['imagenum'] = entry[3]
        if 'videonum' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['videonum'] = entry[4]
        if 'unknownnum' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['unknownnum'] = entry[5]
        if 'textbytes' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['textbytes'] = entry[6]
        if 'imagebytes' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['imagebytes'] = entry[7]
        if 'videobytes' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['videobytes'] = entry[8]
        if 'unknownbytes' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['unknownbytes'] = entry[9]
        if 'ts' not in icp_feature_dict_prelim[icp]:
            icp_feature_dict_prelim[icp]['ts'] = entry[11]
            
        break

for icp, features in icp_feature_dict_prelim.iteritems():
    if 'industry' not in features: print icp
    if 'textnum' not in features: print icp
    if 'imagenum' not in features: print icp
    if 'videonum' not in features: print icp
    if 'unknownnum' not in features: print icp
    if 'textbytes' not in features: print icp
    if 'imagebytes' not in features: print icp
    if 'videobytes' not in features: print icp
    if 'unknownbytes' not in features: print icp

icp_feature_dict_prelim['www.qq.com']

# import pickle
# with open('20170703-icp-features-prelim.pkl', 'wb') as output:
#     pickle.dump(icp_feature_dict, output, -1)

icp_feature_dict_final = {
                          icp:dict(
                                icp=icp,
                                industry=features['industry'],
                                text_bin=text_bin(features['textbytes']),
                                image_bin=image_bin(features['imagebytes']),
                                video_bin=video_bin(features['videobytes'])
                          )
                          for icp, features in icp_feature_dict_prelim.iteritems()
}

# For testing purposes
video_heavy = [cdn for cdn, feature in icp_feature_dict_final.iteritems() if feature['video_bin'] == 'q4']
print len(video_heavy)
print video_heavy

# Create a list of ICP feature dicts
# Ordered by uidx (user/ICP index)
icp_feature_list = [icp_feature_dict_final[uidx_to_icp[uidx]] for uidx in range(num_icps)]

icp_feature_list[0:5]

# Make sure indices match up
uidx_to_icp[1]

# Vectorize! (One-hot encodings of each ICP)
from sklearn.feature_extraction import DictVectorizer
icp_vectorizer = DictVectorizer()
icp_feature_vectors = icp_vectorizer.fit_transform(icp_feature_list)

icp_feature_vectors.shape

1172-1151
# 6 Industries
# 3 x 5 content-type bins

icp_feature_vectors

import pickle
with open('20170703-icp-feature-vectors.pkl', 'w') as output:
    pickle.dump(icp_feature_vectors, output, -1)



