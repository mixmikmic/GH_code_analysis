import numpy as np
import pandas as pd
import scipy.sparse as sp
import csv

import pickle

interactions_pickle = '20170629-interactions-mappings.pkl'
with open(interactions_pickle, 'rb') as output:
    (interactions, iidx_to_cdn, cdn_to_iidx, uidx_to_icp, icp_to_uidx) = pickle.load(output)

# Read in cdn_ips file
cdn_ips_filepath = 'CDN_ips/all_cdn_ips.txt'
cdn_ips_header = ['cdn', 'location', 'isp', 'ips', 'ts']
cdn_ips_datatypes = {
    'cdn': str,
    'location': str,
    'isp': str,
    'ips': np.int64,
    'ts': str
}

cdn_ips_df = pd.read_csv(cdn_ips_filepath, 
                              sep=',', header=None, 
                              names=cdn_ips_header,
                              dtype=cdn_ips_datatypes)

cdn_ips_df.head()

# Read in cdn_ip_locations file
cdn_ip_locations_filepath = 'CDN_IP_Locations/all_cdn_ip_locations.txt'
cdn_ip_locations_header = ['cdn', 'ip', 'location', 'isp', 'ts']
cdn_ip_locations_dtypes = {
    'cdn': str,
    'ip': str,
    'location': str,
    'isp': str,
    'ts': str
}

cdn_ip_locations_df = pd.read_csv(cdn_ip_locations_filepath, 
                              sep=',', header=None, 
                              names=cdn_ip_locations_header,
                              dtype=cdn_ip_locations_dtypes)

# Get no. unique IPs for each CDN
cdn_num_ips = cdn_ip_locations_df.groupby('cdn')['ip'].nunique()
cdn_num_ips

# No. CDNs represented in each file
print cdn_ips_df['cdn'].nunique()
print cdn_ip_locations_df['cdn'].nunique()
# https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas

# Create CDN feature dict with CDN type (encoded in 1st digit of CDN code)
# 0: free
# 1: self-built
# 2: commercial
cdn_feature_dict = { cdn:dict(type=cdn[0]) for cdn in cdn_to_iidx }
len(cdn_feature_dict)

cdn_feature_dict['201']

cdn_temp_dns_filepath = '~/Desktop/CDN Data/cdn_temp_dns/000009_0'
cdn_temp_dns_header = ['cdn', 'isp', 'cname', 'location', 'delay', 'time', 'ip', 'ts']
cdn_temp_dns_dtypes = {
    'cdn': str,
    'isp': str,
    'cname': str,
    'location': str,
    'delay': str,
    'time': str,
    'ip': str,
    'ts': str
}

cdn_temp_dns = pd.read_csv(cdn_temp_dns_filepath, 
                              sep=',', header=None, 
                              names=cdn_temp_dns_header,
                              dtype=cdn_temp_dns_dtypes)

cdn_temp_dns.head()
# Looked at .csv file
# Looks like there can be multiple 'ip's listed before 'ts'

# Appears to be same CDN list as in CDN_ip_locations!
# Tested for all cdn_temp_dns files --> confirmed
print cdn_temp_dns['cdn'].nunique()
cdn_temp_dns.groupby('cdn')['ip'].nunique()

cdn_temp_qos_filepath = '~/Desktop/CDN Data/cdn_temp_qos/000004_0'
cdn_temp_qos_header = ['cdn', 'cdnip', 'location', 'isp', 'tcp', 'ft',                      'mt', 'faultFlag', 'rc', 'url', 'dns', 'ssl', 'dt',                      'tt', 'ds', 'avg', 'max', 'min', 'loss',                      'ip', 'cname', 'ts']

cdn_temp_qos = pd.read_csv(cdn_temp_qos_filepath,
                          sep=',', header=None,
                          names=cdn_temp_qos_header,
                          dtype=str)

print cdn_temp_qos['cdn'].nunique()
print sorted(cdn_temp_qos['cdn'].unique())
# Missing CDN 008, 013

cdn_temp_qos.head()

cdn_temp_qos.groupby('cdn')['cdnip'].nunique()

cdn_num_ips_df = cdn_num_ips.to_frame()

cdn_num_ips_df['ip'].quantile(1)

num_ips_q1 = 0
num_ips_q2 = cdn_num_ips_df['ip'].quantile(0.25)
num_ips_q3 = cdn_num_ips_df['ip'].quantile(0.50)
num_ips_q4 = cdn_num_ips_df['ip'].quantile(0.75)

def num_ips_bin(num_ips):
    if num_ips >= num_ips_q1 and num_ips < num_ips_q2: return 'q1'
    elif num_ips >= num_ips_q2 and num_ips < num_ips_q3: return 'q2'
    elif num_ips >= num_ips_q3 and num_ips < num_ips_q4: return 'q3'
    elif num_ips >= num_ips_q4: return 'q4'
    else: return 'Error'

for entry in cdn_num_ips_df.itertuples():
    cdn = entry[0]
    num_ips = entry[1]
    ips_bin = num_ips_bin(num_ips)
    
    if cdn in cdn_to_iidx:
        cdn_feature_dict[cdn]['num_ips_bin'] = ips_bin

for cdn, features in cdn_feature_dict.iteritems():
    features['cdn'] = cdn

cdn_feature_dict

# Create a list of CDN feature dicts
# Ordered by iidx (item/CDN index)
cdn_feature_list = [cdn_feature_dict[iidx_to_cdn[iidx]] for iidx in range(len(iidx_to_cdn))]

print cdn_feature_list[0]
print cdn_feature_list[1]
print iidx_to_cdn[0]
print iidx_to_cdn[1]

# Vectorize! (One-hot encodings of each ICP)
from sklearn.feature_extraction import DictVectorizer
cdn_vectorizer = DictVectorizer()
cdn_feature_vectors = cdn_vectorizer.fit_transform(cdn_feature_list)

cdn_feature_vectors
# 7 extra features: 4 bins + 3 types

import pickle
with open('20170703-cdn-feature-vectors.pkl', 'w') as output:
    pickle.dump(cdn_feature_vectors, output, -1)



