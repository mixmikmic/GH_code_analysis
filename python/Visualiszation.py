import pandas as pd
import logging
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os as os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os as os
import sys
import csv

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(500 * 1024 * 1024)

data_2010_2017 = pd.read_csv("data_2010_2017_61717.csv",engine='python')

total_2010_2017 = pd.concat([data_2010_2017,data_2012],axis=0)

amazon2010_2017 = total_2010_2017[total_2010_2017["Name"].str.contains("amazon")]

total_2010_2017["Name"].value_counts()

altria_2010_2017 = total_2010_2017[total_2010_2017["Name"].str.contains("altria group inc")]

amazon_altria = pd.concat([amazon2010_2017, altria_2010_2017],axis=0).reset_index(drop=True)

amazon_altria.to_csv("amazon_altria.csv",index=False)

data_2010_2017.head(2)

data_2012 = pd.read_csv("data_file_61717.csv")

data_2012 = data_2012[["CIK #","Name","company","report","date","url","full_text","year","quarter","report_type","status"]]

Amazon_2012 = data_2012[data_2012["Name"].str.contains("amazon")].reset_index(drop = True)

Amazon = data_2010_2017[data_2010_2017["Name"].str.contains("amazon")].reset_index(drop = True)

amazon = pd.concat([Amazon_2012, Amazon],axis=0)

amazon.columns

total_2010_2017.head(1)

total_2010_2017.to_csv("final_data_2012_2017.csv",index = False)

# MyDocs reading from a data frame
class MyDocs(object):
    def __iter__(self):
        for i in range(total_2010_2017.shape[0]):
            yield TaggedDocument(words=gensim.utils.simple_preprocess(total_2010_2017.iloc[i,6]), tags=['%s' % total_2010_2017.iloc[i,1]])

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

get_ipython().run_cell_magic('time', '', '\nif not os.path.exists(\'models/doc2vec.model_new1\'):\n    print "start traing doc2vec model..."\n    documents = MyDocs()\n    doc2vec_model = Doc2Vec(dm=1, dbow_words=1, size=200, window=8, min_count=20, workers=1)\n    doc2vec_model.build_vocab(documents)\n    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)\n    if not os.path.exists(\'models\'):\n        os.makedirs(\'models\')\n        doc2vec_model.save(\'models/doc2vec.model_new1\')\n    else:\n        doc2vec_model.save(\'models/doc2vec.model_new1\')\nelse:\n    doc2vec_model = Doc2Vec.load(\'models/doc2vec.model_new1\')')

doc2vec_model.docvecs.most_similar('netflix com inc', topn=20)

data_2010_2017[data_2010_2017["Name"].str.contains("netflix")]

docvec = doc2vec_model.docvecs[4]

docvec

data_2012["status"].value_counts()

data_2010_2017[data_2010_2017["status"]=="loser"].head(40)

480./(480+147)

147./(480+147)

pca_data = pd.read_csv("Doc_vec.csv")

del pca_data['Unnamed: 0']

x = pca

x.head(1)

pca_data

feature = x.iloc[ :,:200]

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(feature)
PCA(copy=True, iterated_power='auto', n_components=50, random_state=None,
svd_solver='auto', tol=0.0, whiten=False)

a=pca.fit_transform(feature)

a.shape

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
b = model.fit_transform(a) 

b.shape

full_data = pd.read_csv("data_file_61717.csv")

full_data = full_data[["CIK #","Name","company","date","quarter","report","status"]]

full_data = full_data.rename(index = str, columns = {"status":"old"})

tsne = pd.DataFrame(b)

tsne=tsne.rename(index=str, columns={0: "x_tsne", 1: "y_tsne"})

tsne = tsne.reset_index(drop= True)

aaa=pd.DataFrame(predict)

aaa = aaa.reset_index(drop=True)

graphing_tsne = pd.concat([tsne, aaa], axis = 1)

graphing_tsne.reset_index(drop = True, inplace=True)
full_data.reset_index(drop=True, inplace= True)

new = pd.concat([graphing_tsne,full_data],axis = 1)

new["CIK #"] = new["CIK #"].apply(lambda y: str(y))

new.head(2)

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
get_ipython().magic('matplotlib inline')

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

sector = pd.read_csv("sector_cik.csv",dtype=object)

sector.head(1)

sector = sector[["CIK #","Ticker","Sector"]]

new_graphing = pd.merge(new, sector, how = "left", left_on = "CIK #", right_on = "CIK #")

new_graphing.columns



import seaborn as sns
sns.set(style="ticks", context="talk")

g = sns.lmplot(x="x_tsne", y="y_tsne", hue="Sector", data=new_graphing, size=7, fit_reg=False)



feature = x.iloc[ :,:200]
predict = x.iloc[:,201]
feature.shape

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(feature)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
svd_solver='auto', tol=0.0, whiten=False)

a=pca.fit_transform(feature)

a

PC = pd.DataFrame(a)

PC = PC.reset_index(drop=True)

graphing = pd.concat([PC, predict],axis=1)

graphing.columns

graphing = graphing.rename(index=str, columns={0: "x", 1: "y"})

graphing_total = pd.concat([graphing.reset_index(drop=True), new_graphing], axis = 1)

graphing_total.columns

graphing_total = graphing_total[["x","y","status","x_tsne","y_tsne","CIK #","Name","company","date","Ticker","Sector"]]

import seaborn as sns
sns.set(style="ticks", context="talk")

g = sns.lmplot(x="x", y="y", hue="Sector", data=graphing_total,size=7, fit_reg=False)


pca_data.head(1)

pca_new = pca_data.rename(index = str, columns={"0":"x","1":"y"})

import seaborn as sns
sns.set(style="ticks", context="talk")

h = sns.lmplot(x="x", y="y", hue="status", data=pca_new, size=7, fit_reg=False)

graphing_total["Sector"].value_counts()

graphing_total.loc[graphing_total["Sector"]=="Health Care","new_sector"] = "Healthcare"
graphing_total.loc[graphing_total["Sector"]=="Healthcare","new_sector"] = "Healthcare"
graphing_total.loc[graphing_total["Sector"]=="Information Technology","new_sector"] = "Tech"
graphing_total.loc[graphing_total["Sector"]=="Technology","new_sector"] = "Tech"
graphing_total.loc[graphing_total["Sector"]=="Financials","new_sector"] = "Finance"
graphing_total.loc[graphing_total["Sector"]=="Financial","new_sector"] = "Finance"
graphing_total.loc[graphing_total["Sector"]=="Industrial Goods","new_sector"] = "Industrials"
graphing_total.loc[graphing_total["Sector"]=="Industrials","new_sector"] = "Industrials"
graphing_total.loc[graphing_total["Sector"]=="Consumer Staples","new_sector"] = "Consumer Staples"
graphing_total.loc[graphing_total["Sector"]=="Consumer Goods","new_sector"] = "Consumer Goods"
graphing_total.loc[graphing_total["Sector"]=="Consumer Discretionary","new_sector"] = "Consumer Discretionary"
graphing_total.loc[graphing_total["Sector"]=="Services","new_sector"] = "Services"
graphing_total.loc[graphing_total["Sector"]=="Basic Materials","new_sector"] = "Materials"
graphing_total.loc[graphing_total["Sector"]=="Materials","new_sector"] = "Materials"
graphing_total.loc[graphing_total["Sector"]=="Energy","new_sector"] = "Energy"
graphing_total.loc[graphing_total["Sector"]=="Utilities","new_sector"] = "Utilities"
graphing_total.loc[graphing_total["Sector"]=="Telecommunications Services","new_sector"] = "Telecommunications Services"

import seaborn as sns
current_palette_12 = sns.color_palette("Set1", n_colors=12, desat=.5)
sns.set(style="ticks", context="talk",palette=current_palette_12)

h = sns.lmplot(x="x", y="y", hue="new_sector", data=graphing_total, size=7, fit_reg=False)

graphing_total['Name'].head(2)

import seaborn as sns
current_palette_12 = sns.color_palette("Set1", n_colors=12, desat=.5)
sns.set(style="ticks", context="talk",palette=current_palette_12)

h = sns.lmplot(x="x_tsne", y="y_tsne", hue="new_sector", data=graphing_total, size=7, fit_reg=False)

#graphing_total[['x_tsne','y_tsne','Name']].apply(lambda x: h.annotate(*Name),axis=1);



