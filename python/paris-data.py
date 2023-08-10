import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

paris_df = pd.read_excel("accessibilite-des-gares-et-stations-metro-et-rer-ratp.xls")

paris_df.head()

paris_df.groupby("Accessibilité UFR").count()

len(paris_df)

round((156/len(paris_df))*100)

non_accessible_df = paris_df[paris_df['Accessibilité UFR'] == 0]

non_accessible_df.head()

non_accessible_df.groupby("Annonce Sonore Prochain Passage").count()

non_accessible_df.groupby("Annonce Visuelle Prochain Passage").count()

non_accessible_df.groupby("Annonce Sonore Situations Perturbées").count()

non_accessible_df.groupby("Annonce Visuelle Situations Perturbées").count()

non_accessible_df.groupby("Departement").count()

round((625/len(non_accessible_df))*100)

round((46/len(non_accessible_df))*100)

round((38/len(non_accessible_df))*100)

round((36/len(non_accessible_df))*100)

non_accessible_df.groupby("Code Stif").count().sort_values(by="Accessibilité UFR", ascending=False)

unkonwn_line = non_accessible_df[non_accessible_df["Code Stif"] == 1001110020001]

unkonwn_line

paris_total = paris_df.groupby("Code Stif").count()

paris_total

accessible_df = paris_df[paris_df['Accessibilité UFR'] == 1]

accessible_df.head()

accesible_total = accessible_df.groupby("Code Stif").count()

accesible_total

new_df = pd.merge(accesible_total, paris_total, left_index=True, right_index=True, how='outer')

final_df = new_df.fillna(0)

final_df

final_df['% of accessible stations per line'] = round((final_df['Accessibilité UFR_x']/final_df['Accessibilité UFR_y']) * 100)

final_df['% of not accessible stations per line'] = 100 - final_df['% of accessible stations per line']

final_df['Line'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'Funicular', '3 bis', '7 bis', 'OrlyVal', 'RER A', 'RER B']

final_df

fig, ax = plt.subplots()

fig.set_facecolor('gainsboro')
ax.set_axis_bgcolor('gainsboro')

final_df.plot(kind='bar', x='Line', y=['% of not accessible stations per line', '% of accessible stations per line'], stacked=True, rot=1, ax=ax, legend=False)

ax.yaxis.grid(which='major', color='grey', linestyle='-', linewidth=0.2)

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
    which='major',
    top='off', 
    left='off',
    right='off',
    bottom='off',
    labelright='off', 
    labeltop='off',
    labelbottom='on')

ax.set_title("How accesible are Paris' subway stations?")
ax.set_xlabel("Subway line")
ax.set_ylabel("% of accessible stations")

plt.savefig("paris-accessible-stations.pdf")



