import pandas
from sklearn.feature_extraction.text import CountVectorizer

#read in data
df = pandas.read_csv("../data/comparativewomensmovement_dataset.csv", sep='\t', index_col=0, encoding='utf-8')
df

#concatenate the documents from each organization together, creaing four strings

redstockings = df[df['org']=='redstockings']
redstockings_string = ' '.join(str(s) for s in redstockings['text_string'].tolist())
cwlu = df[df['org']=='cwlu']
cwlu_string = ' '.join(str(s) for s in cwlu['text_string'].tolist())
heterodoxy = df[df['org']=='heterodoxy']
heterodoxy_string = ' '.join(str(s) for s in heterodoxy['text_string'].tolist())
hullhouse = df[df['org']=='hullhouse']
hullhouse_string = ' '.join(str(s) for s in hullhouse['text_string'].tolist())

#initialize countvectorizer function, removing stop words
countvec = CountVectorizer(stop_words="english")

redstockings_cwlu = pandas.DataFrame(countvec.fit_transform([redstockings_string, cwlu_string]).toarray(), columns=countvec.get_feature_names())
redstockings_cwlu['word_count'] = redstockings_cwlu.sum(axis=1)
redstockings_cwlu = redstockings_cwlu.iloc[:,0:].div(redstockings_cwlu.word_count, axis=0)
redstockings_cwlu.loc[2] = redstockings_cwlu.loc[0] - redstockings_cwlu.loc[1]
#The words with the highest difference of proportions are distinct to Redstocking
#The words with the lowest (the highest negative) difference of proportions are distinct to CWLU
redstockings_cwlu.loc[2].sort_values(axis=0, ascending=False)

#Heterodoxy versus Hull House
heterodoxy_hullhouse = pandas.DataFrame(countvec.fit_transform([heterodoxy_string, hullhouse_string]).toarray(), columns=countvec.get_feature_names())
heterodoxy_hullhouse['word_count'] = heterodoxy_hullhouse.sum(axis=1)
heterodoxy_hullhouse = heterodoxy_hullhouse.iloc[:,0:].div(heterodoxy_hullhouse.word_count, axis=0)
heterodoxy_hullhouse.loc[2] = heterodoxy_hullhouse.loc[0] - heterodoxy_hullhouse.loc[1]

#The words with the highest difference of proportions are distinct to Heterodoxy
#The words with the lowest (the highest negative) difference of proportions are distinct to Hull House
heterodoxy_hullhouse.loc[2].sort_values(axis=0, ascending=False)

