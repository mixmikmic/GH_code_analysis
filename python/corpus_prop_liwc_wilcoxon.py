import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# load corpus
corpus = pd.read_csv('corpus.csv.gz', compression='gzip')
corpus = corpus[corpus['qual_a_melhor_classificao_para_esse_texto:confidence'] == 1]
corpus = corpus.reset_index()

corpus_feat = pd.read_csv('corpus_liwc_mtx.csv.gz', compression='gzip')
corpus_feat.shape

import re

def wc(x): 
    try:
        return len(re.findall(r'\w+', x['content']))
    except:
        return 0
    
corpus['wc'] = corpus.apply(wc,axis=1)
corpus_feat['wc'] = corpus['wc']

corpus_feat.drop('Unnamed: 0', axis=1,inplace=True)
corpus_feat.drop('confidence', axis=1,inplace=True)

wc_vector = corpus_feat['wc']
class_vector = corpus_feat['class']

corpus_feat.drop('class',axis=1,inplace=True)
corpus_feat.drop('wc',axis=1,inplace=True)

data = corpus_feat.as_matrix().astype(float) / wc_vector.as_matrix().astype(float)[:, np.newaxis]
data[np.isnan(data)] = 0
data[data >= 1E308] = 0
data.shape

columns = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number','swear','social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']
prop_liwc = pd.DataFrame(data, columns=columns)
prop_liwc['class'] = class_vector

outro = prop_liwc[prop_liwc['class'] == 'outro']
diario = prop_liwc[prop_liwc['class'] == 'diario']

def diff(x): 
    return (x['diario_mean']) - (x['outro_mean'])
    #return 100 * ((x['diario_mean']/diario.wc.mean()) - (x['outro_mean']/outro.wc.mean()))
    
stats = pd.DataFrame(data={'diario_mean': diario.mean(axis=0)}, index=columns)
stats['diario_std'] = diario.std(axis=0)
stats['outro_mean'] = outro.mean(axis=0)
stats['outro_std'] = outro.std(axis=0)
stats['diff'] = stats.apply(diff,axis=1)
stats = stats * 100

outro = prop_liwc[prop_liwc['class'] == 'outro']
diario = prop_liwc[prop_liwc['class'] == 'diario'].sample(len(outro))

significance = []
for column in list(stats.index.values):
    a = diario[column]
    b = outro[column]
    t, p = wilcoxon(a, b)
    significance.append(p)
stats['significance'] = significance

linguistic_columns = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number']
linguistic_stats = stats.ix[linguistic_columns]
linguistic_stats.sort_values('diff',ascending=False)
linguistic_stats[linguistic_stats.significance <= 0.05].sort_values('significance',ascending=True)

linguistic_stats[linguistic_stats.significance > 0.05].sort_values('significance',ascending=False)

psychological_columns = ['swear','social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']
psychoProc_stats = stats.ix[psychological_columns]
psychoProc_stats[psychoProc_stats.significance <= 0.05].sort_values('significance',ascending=True).head(10)

psychoProc_stats[psychoProc_stats.significance > 0.05].sort_values('significance',ascending=False).head(10)



