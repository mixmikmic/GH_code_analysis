get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import seaborn as sns
sns.set_context("talk")

df = pd.read_excel('../data/WPP2015_FERT_F04_TOTAL_FERTILITY.XLS', skiprows=16, index_col = 'Country code')
df = df[df.index < 900]  # codes 900+ are regions, not countries

df.rename(columns={df.columns[2]:'Description'}, inplace=True)

df.drop(df.columns[[0, 1, 3, 16]], axis=1, inplace=True) # drop what we dont need

df.sort_values(by='2005-2010', ascending=True, inplace=True)

df.head()

X = df.drop(df.columns[0], axis=1)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

tsne = TSNE(2, random_state=1)

data_2d = tsne.fit_transform(X)  
x = data_2d[:,0]
y = data_2d[:,1]

colors = [i for i,d in enumerate(df['Description'])]
fig, ax = plt.subplots()
scatter = ax.scatter(x,y,c=colors,cmap='viridis')
tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=[str(i) for i in df.Description])
mpld3.plugins.connect(fig, tooltip)
mpld3.display()

highlight_countries = [
    'Niger', 'Yemen', 'India', 'Brazil', 'Norway', 'France', 'Sweden',
    'United Kingdom', 'Spain', 'Italy', 'Germany', 'Japan', 'China'
]

def figure_16(df, highlight_countries):
    # Subset only countries to highlight, transpose for timeseries
    df_high = df[df.Description.isin(highlight_countries)].T[1:]

    # Subset the rest of the countries, transpose for timeseries
    df_bg = df[~df.Description.isin(highlight_countries)].T[1:]

    # background
    ax = df_bg.plot(legend=False, color='k', alpha=0.04, figsize=(14,10))
    ax.xaxis.tick_top()

    # highlighted countries
    df_high.plot(legend=False, ax=ax)

    # replacement level line
    ax.hlines(y=2.1, xmin=0, xmax=12, color='k', alpha=1, linestyle='dashed')

    # Average over time for all countries
    world_avg = df.mean()
    world_avg.plot(ax=ax, color='k')
    ax.text(11.2,world_avg[-1]+0.1,'World average')

    # labels for highlighted countries on the right side
    for i, country in enumerate(highlight_countries):
        ax.text(11.2,df[df.Description==country].values[0][12],country, alpha=0.5)

    # start y axis at 1
    ax.set_ylim(ymin=0.5)
    return ax

figure_16(df, highlight_countries);

clf = IsolationForest(contamination=0.03,random_state=1234)

clf.fit(X)

pred = clf.predict(X)

highlight_countries = list(df[pred==-1].Description)

figure_16(df, highlight_countries);

