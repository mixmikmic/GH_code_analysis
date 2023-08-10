import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

class Trace:
    def __init__(self, trace_prefix, type):
        self._trace_prefix=trace_prefix
        self._type=type
    
    @property
    def trace_prefix(self):
        return self._trace_prefix
    
    @property
    def type(self):
        return self._type

class ApplicationType:
    STREAMING_VIDEO='Streaming Video'
    VOIP='VoIP'
    APP_DOWNLOAD='Application Downloads'
    ONLINE_GAME='Online Game'

class FeatureLevel:
    FLOW='flow'
    PACKET='pkt'

class Protocol:
    UDP='udp'
    TCP='tcp'

LOCAL_DATA_PATH='data'
FEATURE_FILE_TYPE = 'csv'

youtube = Trace(trace_prefix='youtube', type=ApplicationType.STREAMING_VIDEO)
bilibili = Trace(trace_prefix='bilibili', type=ApplicationType.STREAMING_VIDEO)
skype = Trace(trace_prefix='Skype_HongKong', type=ApplicationType.VOIP)
wechat = Trace(trace_prefix='wechat_video', type=ApplicationType.VOIP)
mac_app_store = Trace(trace_prefix='APP_DOWNLOAD', type=ApplicationType.APP_DOWNLOAD)
google_drive = Trace(trace_prefix='google_drive_download', type=ApplicationType.APP_DOWNLOAD)
lol = Trace(trace_prefix='LOL_AI', type=ApplicationType.ONLINE_GAME)
netease = Trace(trace_prefix='netease_game', type=ApplicationType.ONLINE_GAME)
tencent = Trace(trace_prefix='tencent_game_na', type=ApplicationType.ONLINE_GAME)

TRACES=[youtube, bilibili, skype, wechat, mac_app_store, google_drive, lol, netease, tencent]

def get_tcp_udp_info(trace_name, feature_level, file_type):
    udp_filename = os.path.join(LOCAL_DATA_PATH, '{trace_name}_{udp}_{feature_level}.{file_type}'
                                .format(trace_name=trace_name, udp=Protocol.UDP, feature_level=feature_level, file_type=file_type))
    tcp_filename = os.path.join(LOCAL_DATA_PATH, '{trace_name}_{tcp}_{feature_level}.{file_type}'
                                .format(trace_name=trace_name, tcp=Protocol.TCP, feature_level=feature_level, file_type=file_type))
    udp_info, tcp_info = pd.read_csv(udp_filename).dropna(axis=0, how='any'), pd.read_csv(tcp_filename).dropna(axis=0, how='any')
    return tcp_info.shape[0], udp_info.shape[0]

def get_app_type_feature(trace, file_type):
    tcp_flow_num, udp_flow_num = get_tcp_udp_info(trace.trace_prefix, FeatureLevel.FLOW, FEATURE_FILE_TYPE)
    category = trace.type
    return {
        'category':category,
        'total number flow': tcp_flow_num+udp_flow_num,
        'tcp flow number': tcp_flow_num,
        'udp flow number': udp_flow_num
    }

flow_per_app_type = pd.DataFrame(columns=['category', 'total number flow', 'tcp flow number', 'udp flow number'])
for trace in TRACES:
    flow_feature = get_app_type_feature(trace, FEATURE_FILE_TYPE)
    flow_per_app_type = flow_per_app_type.append(flow_feature, ignore_index=True)
        
flow_per_app_type = flow_per_app_type.groupby('category').sum().reset_index()
flow_per_app_type[['total number flow','tcp flow number','udp flow number']] = flow_per_app_type[['total number flow','tcp flow number','udp flow number']].astype(int)
flow_per_app_type



