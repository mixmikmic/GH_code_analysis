import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

y_test = np.load('../data/y_test.npy', mmap_mode = 'r')
test_preds = np.load('../data/preds/stacker_test_preds.npy', mmap_mode = 'r')

unique_labels = [11,  1,  7, 12,  9, 18,  3, 14, 20, 15,  5,  6,  2,  0,  4, 17, 13, 10,  8, 22, 21, 19, 16]
unique_labelnames = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune',
       'smurf', 'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep',
       'land', 'ftp_write', 'back', 'imap', 'satan', 'phf', 'nmap',
       'multihop', 'warezmaster', 'warezclient', 'spy', 'rootkit']
labelnames = {unique_labels[i]:unique_labelnames[i] for i in range(len(unique_labels))}

bin_link_values_normal = {(x,0):np.sum((y_test==11) & (np.argmax(test_preds, axis = 1)==x)) for x in range(len(unique_labels))}
bin_link_values_malicious = {(x,1):np.sum((y_test!=11) & (np.argmax(test_preds, axis = 1)==x)) for x in range(len(unique_labels))}

bin_link_values_dict = {**bin_link_values_normal, **bin_link_values_malicious}

bin_link_values = list([x[1] for x in bin_link_values_dict.items()])
bin_node_labels = ['normal', 'malicious']+[labelnames[x] for x in range(len(labelnames))]
bin_link_sources = [x[0][0]+2 for x in bin_link_values_dict.items()]
bin_link_targets = [x[0][1] for x in list(bin_link_values_dict.items())]
bin_link_labels = ['' for x in range(len(bin_link_sources))]

data_trace = dict(
    type='sankey',
    width = 1118,
    height = 772,
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "v",
    valueformat = ".0f",
    valuesuffix = "",
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label =  bin_node_labels,
      color =  ['rgba(0,0,0,.9)' for x in bin_node_labels]
    ),
    link = dict(
      source =  bin_link_sources,
      target =  bin_link_targets,
      value =  bin_link_values,
      label =  bin_link_labels
  ))

layout =  dict(
    title = "Attack types flowing into maliciousness prediction",
    font = dict(
      size = 10
    )
)

fig = dict(data=[data_trace], layout=layout)
py.iplot(fig, validate=False)

tp = sum([bin_link_values_dict[(x,1)] for x in range(len(unique_labels)) if x!=11])
fn = sum([bin_link_values_dict[(x,0)] for x in range(len(unique_labels)) if x!=11])
tn = bin_link_values_dict[(11,0)]
fp = bin_link_values_dict[(11,1)]

prec = tp / (tp + fp)
recall = tp / (tp + fn)

print("Precision: ", prec)
print("Recall: ", recall)

bin_link_values_file = open('../data/bin_link_values.csv','w')
bin_node_labels_file = open('../data/bin_node_labels.csv','w')
bin_link_sources_file = open('../data/bin_link_sources.csv','w')
bin_link_targets_file = open('../data/bin_link_targets.csv','w')
bin_link_labels_file = open('../data/bin_link_labels.csv','w')
files_dict = {'../data/bin_link_values.csv':[bin_link_values_file, bin_link_values], 
              '../data/bin_node_labels.csv':[bin_node_labels_file, bin_node_labels], 
              '../data/bin_link_sources.csv':[bin_link_sources_file, bin_link_sources],
              '../data/bin_link_targets.csv':[bin_link_targets_file, bin_link_targets],
              '../data/bin_link_labels.csv':[bin_link_labels_file, bin_link_labels]}
for x in files_dict.keys():
    for y in files_dict[x][1]:
        files_dict[x][0].write("{}\n".format(y))
        files_dict[x][0].flush()

