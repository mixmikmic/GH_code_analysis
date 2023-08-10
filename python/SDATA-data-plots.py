#Force in-line plotting

get_ipython().magic('matplotlib inline')

#Import plotting libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json

### CHANGE THIS FILE PATH TO POINT TO THE JSON FILE ###

FILE_PATH = '/PATH_TO_YOUR_JSON_FILE/material_synthesis_data.json'

#Load the JSON file provided in the article.
raw_json = json.loads(open(FILE_PATH, 'rb').read())

#Now we load the JSON file into a Python object

processed_json = []
for record in raw_json:
    processed_json.append(json.loads(record))

#Total number of papers

print sum(row['num_papers'] for row in processed_json)

#Breakdown of papers per material 

for row in processed_json:
    print row['name'], row['num_papers']

#We compile the computed (via Latent Dirichlet Allocation) topic distributions into a spreadsheet

topic_table = []
for row in processed_json:
    topic_table_row = {}
    topic_table_row['Material System'] = row['name']
    for topic in row['topics']:
        topic_table_row[topic] = row['topics'][topic]
    topic_table.append(topic_table_row)
    
topic_df = pd.DataFrame(topic_table)
topic_df = topic_df.set_index('Material System')
topic_df = topic_df.fillna(0)

#What does the topic distribution look like?
topic_df

#Now we display a heatmap figure of the topic distributions

plt.figure(figsize=(24, 14))
sns.set_style('white')
sns.set_context("poster")
sns.heatmap(topic_df.transpose().sample(n=30, axis=0), cmap="Reds", cbar_kws={"label":"Normalized Frequency"})
plt.tight_layout()

#We pick the second record in the JSON file (since it starts as a 0-indexed array)

active_key = 1

#Print out the name and data fields

print processed_json[active_key]['name']
print processed_json[active_key].keys()

#Generate a plot for hydrothermal and calcination conditions

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(
    processed_json[active_key]['calcine_kde']['temperatures'], 
    processed_json[active_key]['calcine_kde']['temperature_frequencies'],
    c='red',
    label='Calcination'
)
ax1.plot(
    processed_json[active_key]['hydrothermal_kde']['temperatures'], 
    processed_json[active_key]['hydrothermal_kde']['temperature_frequencies'],
    c='blue',
    ls='--',
    label='Hydrothermal'
)
ax1.set_ylabel('Normalized Occurrences')
ax1.set_xlabel('Temperature (C)')
ax1.legend()

ax2.plot(
    processed_json[active_key]['calcine_kde']['times'], 
    processed_json[active_key]['calcine_kde']['time_frequencies'],
    c='red',
    label='Calcination'
)
ax2.plot(
    processed_json[active_key]['hydrothermal_kde']['times'], 
    processed_json[active_key]['hydrothermal_kde']['time_frequencies'],
    c='blue',
    ls='--',
    label='Hydrothermal'
)
ax2.set_ylabel('Normalized Occurrences')
ax2.set_xlabel('Time (h)')
ax2.legend()

#We make a spreadsheet for the associated materials (i.e. co-occurring in synthesis with a target synthesized material)

topic_table = []
for row in processed_json:
    topic_table_row = {}
    topic_table_row['Material System'] = row['name']
    for topic in row['associated_materials']:
        topic_table_row[topic] = row['associated_materials'][topic]
    topic_table.append(topic_table_row)
    
topic_df = pd.DataFrame(topic_table)
topic_df = topic_df.set_index('Material System')
topic_df = topic_df.fillna(0)

#We plot the spreadsheet as a heatmap. Self-counts (exact string matches) show up as zero by default in the raw data,
#(otherwise self-counts would trivially dominate and wash-out all other occurrence counts)
#We also choose a random subsample along the Y-axis, because otherwise this heatmap would be too large to practically read.

plt.figure(figsize=(24, 14))
plt.xticks(rotation=90)
sns.set_style('white')
sns.set_context("poster")
sns.heatmap(topic_df.transpose().sample(n=30, axis=0), cmap="Reds", cbar_kws={"label":"Normalized Frequency"})

rule_acc = 0.77619893 #Accuracy of the baseline neural net based on deterministic rule-applied labels
rule_f1 = 0.65672792 #F1 score of the same as above
 
tc_toks = [10, 100, 1000, 2500, 5253] # Number of labelled tokens used in training the human-trained neural net

tc_accs = [0.2, 0.69, 0.809, 0.85079929, 0.86204855] #Accuracy of the human-trained neural net
tc_f1s = [0, 0.60571425, 0.74980774, 0.79588626, 0.81081312] #F1 score of the same as above

#Below, we use calls to the Matplotlib library to generate a line plot of the training curves

plt.xlabel('Number of Training Words')
plt.ylabel('Evaluation Metric')
plt.plot(tc_toks, tc_accs, label='Accuracy', c='red', lw=2)
plt.plot(tc_toks, tc_f1s, label='F1', c='blue', lw=2)
plt.plot(tc_toks, [rule_acc]*5, label='Baseline Accuracy', c='red', lw=1, ls='--')
plt.plot(tc_toks, [rule_f1]*5, label='Baseline F1', c='blue', lw=1, ls='--')
plt.xlim(-500, 5500)
plt.legend(loc=4)

