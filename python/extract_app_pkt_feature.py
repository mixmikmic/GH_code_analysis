import numpy as np
import pandas as pd
from importlib import reload

# define constants
TRACE_FILE_NAME = 'data/tencent_game_na.pcapng' # replace with your raw trace .pcapng/.pcap file name
TRACE_TCP_PACKET_FEATURE_FILE_NAME = 'data/tencent_game_na_tcp_pkt.csv' # replace with your favorite tcp packet feature .csv file name
TRACE_UDP_PACKET_FEATURE_FILE_NAME = 'data/tencent_game_na_udp_pkt.csv' # replace with your favorite udp packet feature .csv file name
TRACE_PACKET_FEATURE_FILE_NAME = 'data/tencent_game_na_pkt.csv' # replace with your favorite packet feature .csv file name
# LOCAL_IP = '172.16.26.207' # your local ip

# convert raw trace to readable udp and tcp packet feature csv file
from python import packet_feature
reload(packet_feature)
get_ipython().run_line_magic('time', 'packet_feature.tcp_generate(TRACE_FILE_NAME,TRACE_TCP_PACKET_FEATURE_FILE_NAME)')

# read in packet feature csv file and do some transformation
import ipaddress
tcp_pkt_feature_df = pd.read_csv(TRACE_TCP_PACKET_FEATURE_FILE_NAME)
filterer = tcp_pkt_feature_df.apply(lambda row:(not pd.isnull(row['ip.src']) and ipaddress.IPv4Address(row['ip.src']).is_global) or (not pd.isnull(row['ip.dst']) and ipaddress.IPv4Address(row['ip.dst']).is_global),axis=1)
tcp_pkt_feature_df = tcp_pkt_feature_df[filterer]
record_num = tcp_pkt_feature_df.shape[0]
tcp_pkt_feature_df['remote_ip'] = tcp_pkt_feature_df.apply(lambda row:row['ip.dst'] if ipaddress.IPv4Address(row['ip.dst']).is_global else row['ip.src'],axis=1) if record_num > 0 else None
tcp_pkt_feature_df['remote_ip2num'] = tcp_pkt_feature_df.apply(lambda row:int(ipaddress.IPv4Address(row['remote_ip'])),axis=1) if record_num > 0 else None
tcp_pkt_feature_df['protocol'] = 'tcp' if record_num > 0 else None
tcp_pkt_feature_df['is_tcp'] = 1 if record_num > 0 else None
tcp_pkt_feature_df['is_udp'] = 0 if record_num > 0 else None
tcp_pkt_feature_df.rename(columns={'tcp.len':'pkt_len'},inplace=True)

# view the shape of the dataset: (number of records, number of features)
tcp_pkt_feature_df.shape

# view the data types for each feature
tcp_pkt_feature_df.dtypes

# view the statistical features of each numerical feature
tcp_pkt_feature_df.describe()

# view the first 5 records
tcp_pkt_feature_df.head()

# convert raw trace to readable udp and tcp packet feature csv file
from python import packet_feature
reload(packet_feature)
get_ipython().run_line_magic('time', 'packet_feature.udp_generate(TRACE_FILE_NAME,TRACE_UDP_PACKET_FEATURE_FILE_NAME,True)')

udp_pkt_feature_df = pd.read_csv(TRACE_UDP_PACKET_FEATURE_FILE_NAME)
udp_pkt_feature_df

# read in packet feature csv file and do some transformation
import ipaddress
def filter_illegal(row):
    try:
        ipaddress.IPv4Address(row['ip.src'])
        ipaddress.IPv4Address(row['ip.dst'])
        return (not pd.isnull(row['ip.src']) and ipaddress.IPv4Address(row['ip.src']).is_global) or (not pd.isnull(row['ip.dst']) and ipaddress.IPv4Address(row['ip.dst']).is_global)
    except ValueError as e:
        print(e)
        return False                                                                                
                                                                                                
udp_pkt_feature_df = pd.read_csv(TRACE_UDP_PACKET_FEATURE_FILE_NAME)
filterer = udp_pkt_feature_df.apply(filter_illegal,axis=1)
udp_pkt_feature_df = udp_pkt_feature_df[filterer]
record_num = udp_pkt_feature_df.shape[0]
udp_pkt_feature_df['remote_ip'] = udp_pkt_feature_df.apply(lambda row:row['ip.dst'] if ipaddress.IPv4Address(row['ip.dst']).is_global else row['ip.src'],axis=1) if record_num > 0 else None
udp_pkt_feature_df['remote_ip2num'] = udp_pkt_feature_df.apply(lambda row:int(ipaddress.IPv4Address(row['remote_ip'])),axis=1) if record_num > 0 else None
udp_pkt_feature_df['protocol'] = 'udp' if record_num > 0 else None
udp_pkt_feature_df['is_tcp'] = 0 if record_num > 0 else None
udp_pkt_feature_df['is_udp'] = 1 if record_num > 0 else None
udp_pkt_feature_df.rename(columns={'udp.length':'pkt_len'},inplace=True)

# view the shape of the dataset: (number of records, number of features)
udp_pkt_feature_df.shape

# view the data types for each feature
udp_pkt_feature_df.dtypes

# view the statistical features of each numerical feature
udp_pkt_feature_df.describe()

# view the first 5 records
udp_pkt_feature_df.head()

# combine dataframes
pkt_feature_df = tcp_pkt_feature_df[['remote_ip2num','is_tcp','is_udp','pkt_len']].append(udp_pkt_feature_df[['remote_ip2num','is_tcp','is_udp','pkt_len']],ignore_index=True)

# shape
pkt_feature_df.shape

###### column types
pkt_feature_df.dtypes

# describe
pkt_feature_df.describe()

# head 5 records
pkt_feature_df.head()

# write to csv
pkt_feature_df.to_csv(TRACE_PACKET_FEATURE_FILE_NAME, index=False)



