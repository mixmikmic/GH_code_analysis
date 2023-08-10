import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
get_ipython().run_line_magic('matplotlib', 'inline')

base = "C:/Users/name/Dropbox/Misc/Data Vis Challenges/April 2018/"
df = pd.read_csv(base + "the-office-lines.csv", header=0)

temp=[]
for i in range(df.shape[0]):
    temp.append(re.sub("[\(\[].*?[\)\]]", "", df['line_text'][i]).lower())
df['line_text'] = temp

temp = []
for i in range(df.shape[0]):
    temp.append(df['speaker'][i].lower())
df['speaker'] = temp

notd = df.loc[df['deleted'] == False]
didd = df.loc[df['deleted'] == True]

main_speaks = ['michael','dwight','jim','pam','ryan', 'andy',
              'robert', 'jan', 'roy', 'stanley', 'kevin',
              'meredith', 'angela', 'oscar', 'phyllis',
              'kelly', 'toby', 'creed', 'darryl','erin',
              'gabe', 'holly', 'nellie', 'clark', 'pete']

speak_words = {}
for i in main_speaks:
    sdf = notd.loc[df['speaker'] == i]
    allw = []
    for j in sdf['line_text']:
        lst = re.findall(r'\b\w+', j)
        allw.extend(lst)
    speak_words[i] = Counter(allw)

occ_matrix = []
for i in main_speaks:
    lst_occ = []
    for j in main_speaks:
        lst_occ.append(speak_words[i][j]/len(speak_words[i]))
    occ_matrix.append(lst_occ)
    

fig = plt.figure(figsize = (14,14))
ax = plt.subplot(111)
heat = sns.heatmap(occ_matrix, square=True, cmap = 'Blues', xticklabels=main_speaks,
                   yticklabels=main_speaks, ax = ax, cbar_kws={"shrink": .82})
ax.tick_params('both', labelsize = 16)
ax.collections[0].colorbar.set_label("Normalized Mentions", rotation = 270, labelpad = 25, size = 16)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=14)



