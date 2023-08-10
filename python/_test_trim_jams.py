get_ipython().magic('matplotlib inline')
import jams
import milsed
import pumpp
import json
import pickle
import os
import numpy as np

# fileid = 'Y-VULyMtKazE_0.000_7.000'
fileid = 'Y---lTs1dxhU_30.000_40.000'
jamfile = os.path.join('/home/js7561/dev/milsed/models/resources/model011/predictions/',
                       fileid + '.jams')
jam = jams.load(jamfile)

jam

jam_trim = jam.trim(0, 5, strict=False)

jam_trim

jam_trim.annotations.search(annotation_tools='dynamic')

d = jam_trim.annotations.search(annotation_tools='reference')

len(d)

type(d)

type(d[0])

import pickle
import numpy as np

pump = pickle.load(open('/home/js7561/dev/milsed/models/resources/pump.pkl', 'rb'))

duration = 10.0

spred = np.zeros(17)
spred[0] = 0.6
spred = np.asarray([spred])
spred

pump['static'].inverse(spred[0], duration=duration)

spred = np.zeros(17)
spred[0] = 0.5
spred = np.asarray([spred])
pump['static'].inverse(spred[0], duration=duration)

spred = np.zeros(17)
spred[0] = 0.4999
spred = np.asarray([spred])
pump['static'].inverse(spred[0], duration=duration)

import jams
import pumpp

jamfile = '/home/js7561/dev/milsed/models/resources/model015/predictions/Y--0w1YA1Hm4_30.000_40.000.jams'
jam = jams.load(jamfile)

jam

ann_s = jam.annotations.search(annotation_tools='static')[0]
ann_d = jam.annotations.search(annotation_tools='dynamic')[0]

df_d = ann_d.to_dataframe()
df_d

df_d['filename'] = 'audio/test_dynamic.wav'
df_d

df_d['start_time'] = df_d.time
df_d

df_d['end_time'] = df_d.time + df_d.duration
df_d

df_d['label'] = df_d['value']
df_d

df_d_ordered = df_d[['filename', 'start_time', 'end_time', 'label']]
df_d_ordered

fid = 'test_static'
# PROCESS STATIC LABELS
df_s = ann_s.to_dataframe()
df_s['filename'] = 'audio/{}.wav'.format(fid)
df_s['start_time'] = df_s.time
df_s['end_time'] = df_s.time + df_s.duration
df_s['label'] = df_s['value']
df_s_ordered = df_s[['filename', 'start_time', 'end_time', 'label']]

df_s_ordered

df_d_append = df_d_ordered.append(df_d_ordered)

df_d_ordered

df_d_append

df_d_append.to_csv('~/test.txt', header=False, index=False, sep='\t')

OUTPUT_PATH = '/home/js7561/dev/milsed/models/resources/'

jamfile = '/home/js7561/dev/milsed/models/resources/model011/predictions_eval/YeyFPHlybqDg_30.000_40.000.jams'
jam = jams.load(jamfile)

jam

ann_s = jam.annotations.search(annotation_tools='static')[0]
ann_s

durfile = os.path.join(OUTPUT_PATH, 'durations.json')
durations = json.load(open(durfile, 'r'))

fid = 'YeyFPHlybqDg_30.000_40.000'
durations[fid]

df_s = ann_s.to_dataframe()
df_s['filename'] = 'audio/{}.wav'.format(fid)
df_s['start_time'] = df_s.time
df_s['end_time'] = df_s.time + df_s.duration
df_s['label'] = df_s['value']
df_s_ordered = df_s[['filename', 'start_time', 'end_time', 'label']]

df_s_ordered

df_s



