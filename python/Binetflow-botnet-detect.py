get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import os
import datetime as dt
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = os.path.join('..','CTU-13-Dataset')
directory = os.fsencode(dataset_path)

files = os.listdir(directory)
sample_file = files[0]

# read the file with pandas
df = pd.read_csv(os.path.join(directory, sample_file).decode('utf-8'), low_memory=False)
#sample_df[sample_df.Label.str.contains("Botnet")].head(10)
# reindex based on StartTime
# but first drop the ns off the time stamps
df['StartTime'] = df['StartTime'].apply(lambda x: x[:19])
df['StartTime'] = pd.to_datetime(df['StartTime'])
df = df.set_index('StartTime')

# replace NaN with a negative port number
df['Dport'] = df['Dport'].fillna('-1')
df['Dport'] = df['Dport'].apply(lambda x: int(x,0))
df['Sport'] = df['Sport'].fillna('-1')
df['Sport'] = df['Sport'].apply(lambda x: int(x,0))

df.head()

from functools import reduce
import ipaddress

def classify_ip(ip):
    '''
    str ip - ip address string to attempt to classify.
    treat ipv6 addresses as N/A
    '''
    try: 
        ip_addr = ipaddress.ip_address(ip)
        if isinstance(ip_addr,ipaddress.IPv6Address):
            return 'ipv6'
        elif isinstance(ip_addr,ipaddress.IPv4Address):
            # split on .
            octs = ip_addr.exploded.split('.')
            if 0 < int(octs[0]) < 127: return 'A'
            elif 127 < int(octs[0]) < 192: return 'B'
            elif 191 < int(octs[0]) < 224: return 'C'
            else: return 'N/A'
    except ValueError:
        return 'N/A'
    
def avg_duration(x):
    return np.average(x)
    
def n_dports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
n_dports_gt1024.__name__ = 'n_dports>1024'

def n_dports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
n_dports_lt1024.__name__ = 'n_dports<1024'

def n_sports_gt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b>1024 else a),x)
n_sports_gt1024.__name__ = 'n_sports>1024'

def n_sports_lt1024(x):
    if x.size == 0: return 0
    return reduce((lambda a,b: a+b if b<1024 else a),x)
n_sports_lt1024.__name__ = 'n_sports<1024'

def label_atk_v_norm(x):
    for l in x:
        if 'Botnet' in l: return 'Attack'
    return 'Normal'
label_atk_v_norm.__name__ = 'label'

def background_flow_count(x):
    count = 0
    for l in x:
        if 'Background' in l: count += 1
    return count

def normal_flow_count(x):
    if x.size == 0: return 0
    count = 0
    for l in x:
        if 'Normal' in l: count += 1
    return count

def n_conn(x):
    return x.size

def n_tcp(x):
    count = 0
    for p in x: 
        if p == 'tcp': count += 1
    return count
    
def n_udp(x):
    count = 0
    for p in x: 
        if p == 'udp': count += 1
    return count
    
def n_icmp(x):
    count = 0
    for p in x: 
        if p == 'icmp': count += 1
    return count

def n_s_a_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'A': count += 1
    return count
    
def n_d_a_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'A': count += 1
    return count

def n_s_b_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'B': count += 1
    return count

def n_d_b_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'A': count += 1
    return count
        
def n_s_c_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'C': count += 1
    return count
    
def n_d_c_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'C': count += 1
    return count
        
def n_s_na_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'N/A': count += 1
    return count
    
def n_d_na_p_address(x):
    count = 0
    for i in x: 
        if classify_ip(i) == 'N/A': count += 1
    return count

def n_ipv6(x):
    count = 0
    for i in x:
        if classify_ip(i) == 'ipv6': count += 1
    return count
    

# The datastructure to hold our feature extraction funcitons, which will
# get applied to each aggregation of the datasets.
extractors = {
    'Label'   : [label_atk_v_norm,
                 background_flow_count,
                 normal_flow_count,
                 n_conn,
                ],
    'Dport'   : [n_dports_gt1024,
                 n_dports_lt1024
                ],
    'Sport'   : [n_sports_gt1024,
                 n_sports_lt1024,
                ],
    'Dur'     : [avg_duration,
                ],
    'SrcAddr' : [n_s_a_p_address,
                 n_s_b_p_address,
                 n_s_c_p_address,
                 n_s_na_p_address,
                ],
    'DstAddr' : [n_d_a_p_address,
                 n_d_b_p_address,
                 n_d_c_p_address,
                 n_d_na_p_address,
                ],
    'Proto'   : [n_tcp,
                 n_icmp,
                 n_udp,
                ],
}
    
#df.groupby(['StartTime', pd.Grouper(freq='1S')]).agg(extractors)
r = df.resample('1S')
n_df = r.agg(extractors)

n_df.columns = n_df.columns.droplevel(0) # get rid of the heirarchical columns
pd.options.display.max_columns = 99
n_df.head()

corr = n_df.corr()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(corr,
            #cmap = sns.diverging_palette(220, 10, as_cmap=True),
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            ax=ax)

# Create a scatter matrix of the aggregated dataframe
# print(n_df.shape)
tmp = n_df.dropna()
print(tmp.shape)
# n_df[n_df.isnull().any(axis=1)]
# There are a ton of features, so let's see if we can choose a few interesting ones to pairplot based on the heat maps
#v = tmp.columns.tolist()[1:]
v = ['avg_duration','n_udp','background_flow_count']
sns.pairplot(tmp, vars=v, hue='label')

