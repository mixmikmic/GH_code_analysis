get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
minimum_distances = []
capture_distance_fn = "/Users/mjohnson/Desktop/Projects/moss_backbone/nuclear/enrichment/distances.txt"
capture_distance = open(capture_distance_fn)
while True:
    line1=capture_distance.readline().rstrip().split()
    line2=capture_distance.readline().rstrip().split()
    if not line2: break
    for line in (line1,line2):
        if not line[2].startswith("Physco"):
            line[2] = "Pleurocarp-{}".format(line[0])   
    try:
        line1[-1]=float(line1[-1])
        line2[-1]=float(line2[-1])
        min_pdist = np.array((line1[-1],line2[-1])).argmin()
    except ValueError:
        continue
    
    if min_pdist:
        minimum_distances.append(line2)
    else:
        minimum_distances.append(line1)
capture_distance.close()    
capture_distance_df = pd.DataFrame.from_records(minimum_distances,columns=["Gene","Species","Target","Distance"])
capture_distance_df["IsOnekp"] = capture_distance_df.Species.str.contains("onekp")

groups = capture_distance_df.groupby("IsOnekp")
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.hist(group.Distance, bins=40,alpha=0.5)
ax.legend(["Captured","OneKp"])
plt.xlabel("Percent Dissimilarity")
plt.ylabel("Number of Sequences")
plt.rcParams['figure.figsize'] = (3, 4)
plt.show()

import seaborn as sns
#Set up a data frame containing only the four divergent species, from both capture and OneKP Data
takakia = capture_distance_df[(capture_distance_df.Species.str.upper().str.contains("TAKAKIA"))]
takakia.Species="Takakia"
diphyscium = capture_distance_df[(capture_distance_df.Species.str.lower().str.contains("diphyscium"))]
diphyscium.Species="Diphyscium"
buxbaumia = capture_distance_df[(capture_distance_df.Species.str.lower().str.contains("buxbaumia"))]
buxbaumia.Species="Buxbaumia"
tetraphis = capture_distance_df[(capture_distance_df.Species.str.lower().str.contains("tetraphis"))]
tetraphis.Species = "Tetraphis"

#Combine the dataframes
divergent_mosses = pd.concat([takakia,diphyscium,buxbaumia,tetraphis])
divergent_mosses.drop(["Gene","Target"],1,inplace=True)
divergent_mosses

#Plot the data
g = sns.FacetGrid(divergent_mosses,col="Species",size=5,aspect=1)
g = g.map(sns.violinplot,"IsOnekp","Distance",bins=20,color=".8",inner=None)
g = g.map(sns.stripplot,"IsOnekp","Distance",jitter=True)

g.savefig("/Users/mjohnson/Desktop/Projects/moss_backbone/nuclear/enrichment/Hybseq_vs_OneKP.svg")

takakia



