import os
import csv
import json
import numpy as np
import socket,struct
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import time as time
import itertools
import pandas as pd

#Unique prefix to make sure my names don't conflict with yours
MY_SUFFIX = "_" + os.getcwd().split('/')[-1] 

print "MY_SUFFIX =",MY_SUFFIX

drop_objects = True

def drop(pattern):    
    map(atk.drop_frames, filter(lambda x: not x.find(pattern) < 0, atk.get_frame_names()))
    map(atk.drop_graphs, filter(lambda x: not x.find(pattern) < 0, atk.get_graph_names()))
    map(atk.drop_models, filter(lambda x: not x.find(pattern) < 0, atk.get_model_names()))

import trustedanalytics as atk
import trustedanalytics.core.admin as admin

print "ATK installation path = %s" % (atk.__path__)

# Make sure you created the credential file before connecting to ATK
# Replace the server uri value with your ATK instance uri


atk.server.uri = 'ENTER URI HERE'
atk.connect(r'myuser-cred.creds')

#Clearing out old frames and graphs:

reference = MY_SUFFIX ### Please ensure each user uses different value here
score = reference

if drop_objects == True:
    drop('network_edges_'+reference)
    drop('network_graph_'+reference)    
    drop('bytes_out_'+score)
    drop('bytes_in_'+score)
    drop('bytes_in_out_'+score)
    drop('ip_summary_frame_'+score)
    drop('svmModel_' + reference)
    admin.drop_stale("24 hours")
    admin.finalize_dropped()

dataset = "TYPE HDFS URI HERE"
week2_nf_schema=[('TimeSeconds', atk.float64),
                 ('tstart', str),
                 ('dateTimeStr', atk.float64),
                 ('protocol', str),
                 ('proto', str),
                 ('sip', str),
                 ('dip', str),
                 ('sport', atk.int64),
                 ('dport', atk.int64),
                 ('flag', atk.int64),
                 ('fwd', atk.int64),
                 ('tdur', atk.int64),
                 ('firstSeenSrcPayloadBytes', atk.int64),
                 ('firstSeenDestPayloadBytes', atk.int64),
                 ('ibyt', atk.int64),
                 ('obyt', atk.int64),
                 ('ipkt', atk.int64),
                 ('opkt', atk.int64),
                 ('recordForceOut', atk.int64)]

real_netflow = atk.Frame(atk.CsvFile(dataset, week2_nf_schema,  skip_header_lines=1))
if drop_objects == True:
    drop('real_netflow_'+reference)
real_netflow.name='real_netflow_'+reference

real_netflow.inspect(wrap=10, round=4, width=100)

def get_time(row):
    x = row.tstart.split(" ")
    date = x[0]
    time = x[1]
    y = time.split(":")
    hour = atk.float64(y[0])
    minute = atk.float64(y[1])
    numeric_time = hour + minute/60.0
    return [date, time, hour, minute, numeric_time]

real_netflow.add_columns(get_time, [('date',str), 
                                    ('time',str), 
                                    ('hour', atk.float64), 
                                    ('minute', atk.float64), 
                                    ('numeric_time', atk.float64)], 
                         columns_accessed=['tstart'])

real_netflow.inspect(wrap=10, round=4, width=100,
                     columns=['date', 'time', 'hour', 'minute', 'numeric_time'])

network_edges = real_netflow.group_by(['sip', 'dip'], atk.agg.count)
if drop_objects == True:
    drop('network_edges_'+reference)
network_edges.name = 'network_edges_'+reference

network_graph = atk.Graph()
if drop_objects == True:
    drop('network_graph_'+reference)
network_graph.name = 'network_graph_'+reference
network_graph.define_vertex_type('IP')
network_graph.define_edge_type('Connection', 'IP', 'IP', directed=False)
network_graph.edges['Connection'].add_edges(network_edges, 'sip', 'dip', ['count'], create_missing_vertices = True)

bytes_out = real_netflow.group_by(['sip'], atk.agg.count, {'ibyt': atk.agg.sum})
bytes_out.rename_columns({'ibyt_SUM': 'obyt_SUM', 'count': 'outgoing_connections'})

if drop_objects == True:
    drop('bytes_out_'+score)
bytes_out.name = 'bytes_out_'+score
bytes_in = real_netflow.group_by(['dip'], {'ibyt': atk.agg.sum})

if drop_objects == True:
    drop('bytes_in_'+score)
bytes_in.name = 'bytes_in_'+score
bytes_in_out = bytes_out.join(bytes_in, left_on='sip', right_on='dip', how='inner')

if drop_objects == True:
    drop('bytes_in_out_'+score)
bytes_in_out.name = 'bytes_in_out_'+score
bytes_in_out.add_columns(lambda row: [np.log(row.ibyt_SUM), np.log(row.obyt_SUM)], 
                         [('ln_ibyt_SUM', atk.float64), ('ln_obyt_SUM', atk.float64)])

bytes_in_out.inspect(wrap=10, round=4, width=100)

print 'Compute degree count:'
unweighted_degree_frame = network_graph.annotate_degrees('degree', 'undirected')['IP']

print 'Compute weighted degree count:'
weighted_degree_frame = network_graph.annotate_weighted_degrees('weighted_degree', 'undirected', 
                                                                      edge_weight_property = 'count')['IP']

ip_summary_frame_intermidiate = bytes_in_out.join(weighted_degree_frame, left_on='sip', right_on='sip', how='inner')
ip_summary_frame = ip_summary_frame_intermidiate.join(unweighted_degree_frame, left_on='sip', right_on='sip', how='inner')

if drop_objects == True:
    drop('ip_summary_frame_'+score)
ip_summary_frame.name = 'ip_summary_frame_'+score
#ip_summary_frame.drop_columns(['dip', '_vid_L', '_vid_R', '_label_L', '_label_R', 'sip_L', 'sip_R'])
ip_summary_frame.add_columns(lambda row: [row.ibyt_SUM + row.obyt_SUM, 
                                          np.log(row.ibyt_SUM + row.obyt_SUM),
                                          np.log(row.degree),
                                          np.log(row.weighted_degree)], 
                             [('total_traffic', atk.float64),
                             ('ln_total_traffic', atk.float64),
                             ('ln_degree', atk.float64),
                             ('ln_weighted_degree', atk.float64)])
ip_summary_frame.rename_columns({'sip': 'ip'})
print ip_summary_frame.inspect(wrap=10, round=4, width=100)
ip_summary_frame_pd = ip_summary_frame.download(ip_summary_frame.row_count)
ip_summary_frame_pd.head()

outgoing_connections_bins = 100
ln_ibyt_SUM_bins = 100
ln_obyt_SUM_bins = 100
ln_total_traffic_bins = 100
weighted_degree_bins = 100
degree_bins = 100

def plot_hist(histogram):
    plt.bar(histogram.cutoffs[:-1], histogram.hist, width = histogram.cutoffs[1]-histogram.cutoffs[0])
    plt.xlim(min(histogram.cutoffs), max(histogram.cutoffs))

histograms = {}
histograms['outgoing_connections'] = ip_summary_frame.histogram(
    column_name = "outgoing_connections", num_bins=outgoing_connections_bins)
histograms['ln_ibyt_SUM'] = ip_summary_frame.histogram(column_name = "ln_ibyt_SUM", num_bins=ln_ibyt_SUM_bins)
histograms['ln_obyt_SUM'] = ip_summary_frame.histogram(column_name = "ln_obyt_SUM", num_bins=ln_obyt_SUM_bins)
histograms['ln_total_traffic'] = ip_summary_frame.histogram(column_name = "ln_total_traffic", num_bins=ln_total_traffic_bins)
histograms['ln_weighted_degree'] = ip_summary_frame.histogram(column_name = "ln_weighted_degree", num_bins=weighted_degree_bins)
histograms['ln_degree'] = ip_summary_frame.histogram(column_name = "ln_degree", num_bins=degree_bins)

plot_hist(histograms['outgoing_connections'])

plot_hist(histograms['ln_ibyt_SUM'])

plot_hist(histograms['ln_obyt_SUM'])

plot_hist(histograms['ln_total_traffic'])

plot_hist(histograms['ln_weighted_degree'])

plot_hist(histograms['ln_degree'])

# import the scatter_matrix functionality
from pandas.tools.plotting import scatter_matrix

# define colors list, to be used to plot survived either red (=0) or green (=1)
colors=['yellow','green']

# make a scatter plot
df = pd.DataFrame(ip_summary_frame.take(ip_summary_frame.row_count, 
                                        columns=["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree"]), 
                  columns=["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree"])

scatter_matrix(df, figsize=[20,20],marker='x',
               c=df.ln_degree.apply(lambda x: colors[1]))

df.info()
print ip_summary_frame.inspect(wrap=10, round=4, width=100)

ip_summary_frame = atk.get_frame('ip_summary_frame_' + reference)
ip_summary_frame.add_columns(lambda row: '1', ('label', atk.float64))

if drop_objects == True:
    drop('svmModel_' + reference)
SVM_model = atk.LibsvmModel('svmModel_' + reference)
SVM_model.train(ip_summary_frame, 'label', 
                ["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree"],
                nu=0.5, svm_type=2)

scored_frame = SVM_model.predict(ip_summary_frame, 
                                 ["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree"])

scored_frame_group_by = scored_frame.group_by("predicted_label", atk.agg.count)
scored_frame_group_by.inspect(wrap=10, round=4, width=100)

df = pd.DataFrame(scored_frame.take(ip_summary_frame.row_count, 
                                    columns=["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree", "predicted_label"]), 
                  columns=["ln_ibyt_SUM", "ln_obyt_SUM", "ln_degree", "ln_weighted_degree", "predicted_label"])

#scored_frame_pd = scored_frame.download(scored_frame.row_count)
# import the scatter_matrix functionality
from pandas.tools.plotting import scatter_matrix

# define colors list, to be used to plot survived either red (=0) or green (=1)
colors=['red','green']

# make a scatter plot
scatter_matrix(df, figsize=[20,20],marker='x',
               c=df.predicted_label.apply(lambda x: (colors[1] if x == 1 else colors[0])))

df.info()

scored_frame.export_to_csv("scored_frame_hour_%s" %(MY_SUFFIX),",",)

atk.get_frame_names()

atk.get_model_names()

SVM_model.publish()



