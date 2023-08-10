import pandas as pd
import os 
import glob
from tqdm import tqdm_notebook, tnrange

from_pipeline = '/projects/ps-yeolab3/bay001/rep_element_reference/example5/RBFOX2/results/ecliprepmap_concatenated.sam'
from_eric = '/projects/ps-yeolab3/bay001/rep_element_reference/example5/eric_reference.sam'

pipeline_length = get_ipython().getoutput('wc -l $from_pipeline')
pipeline_length = int(pipeline_length[0].split(' ')[0])

from_eric_length = get_ipython().getoutput('wc -l $from_eric')
from_eric_length = int(from_eric_length[0].split(' ')[0])

print(pipeline_length - from_eric_length)

# first gather every mapped transcript
transcripts = []
progress = tnrange(pipeline_length)
with open(from_pipeline, 'r') as f:
    for line in f:
        transcripts.append(line.split('\t')[2])
        progress.update(1)
transcripts = set(transcripts)

# first gather every mapped transcript from eric
erics_transcripts = []
progress = tnrange(from_eric_length)
with open(from_eric, 'r') as f:
    for line in f:
        erics_transcripts.append(line.split('\t')[2])
        progress.update(1)
erics_transcripts = set(erics_transcripts)

len(transcripts - erics_transcripts)

len(erics_transcripts - transcripts)

len(erics_transcripts)

# first gather every mapped transcript
transcripts = []
progress = tnrange(pipeline_length)
with open(from_pipeline, 'r') as f:
    for line in f:
        transcripts.append("{}:{}".format(line.split('\t')[2],line.split('\t')[3]))
        progress.update(1)
transcripts = set(transcripts)

# first gather every mapped transcript from eric
erics_transcripts = []
progress = tnrange(from_eric_length)
with open(from_eric, 'r') as f:
    for line in f:
        erics_transcripts.append("{}:{}".format(line.split('\t')[2],line.split('\t')[3]))
        progress.update(1)
erics_transcripts = set(erics_transcripts)

diff = len(transcripts - erics_transcripts)
print(diff)
diff = len(erics_transcripts - transcripts)
print(diff)

diff/float(len(erics_transcripts))

repCount = []
for different in list(transcripts - erics_transcripts):
    if not different.startswith('chr'):
        repCount.append(different)
print(repCount)



