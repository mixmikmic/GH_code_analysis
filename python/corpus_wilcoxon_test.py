import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
corpus = pd.read_csv('corpus.csv.gz', compression='gzip')
corpus_liwc = pd.read_csv('corpus_liwc_mtx.csv.gz', compression='gzip')

corpus_liwc['class'] = corpus['qual_a_melhor_classificao_para_esse_texto']
corpus_liwc['confidence'] = corpus['qual_a_melhor_classificao_para_esse_texto:confidence']

import re

def wc(x): 
    try:
        return len(re.findall(r'\w+', x['content']))
    except:
        return 0
    
corpus['wc'] = corpus.apply(wc,axis=1)
corpus_liwc['wc'] = corpus['wc']
corpus_liwc = corpus_liwc[corpus_liwc['confidence'] == 1]
corpus_liwc[['class','wc']].groupby(['class']).agg(['mean','count'])

outro = corpus_liwc[corpus_liwc['class'] == 'outro']
diario = corpus_liwc[corpus_liwc['class'] == 'diario'].sample(len(outro))

columns = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number','swear','social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']

def diff(x): 
    return (x['diario_mean']) - (x['outro_mean'])
    #return 100 * ((x['diario_mean']/diario.wc.mean()) - (x['outro_mean']/outro.wc.mean()))
    
stats = pd.DataFrame(data={'diario_mean': diario.mean(axis=0)}, index=columns)
stats['outro_mean'] = outro.mean(axis=0)

stats['diff'] = stats.apply(diff,axis=1)

significance = []
for column in list(stats.index.values):
    a = diario[column]
    b = outro[column]
    t, p = wilcoxon(a, b)
    significance.append(p)
stats['significance'] = significance

liguistic_columns = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number']
liguistic_stats = stats.ix[liguistic_columns]
liguistic_stats[['significance','diff']][liguistic_stats.significance > 0.05].sort_values('significance',ascending=False)

psychological_columns = ['swear','social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']
psychological_stats = stats.ix[psychological_columns]
psychological_stats[['significance','diff']][psychological_stats.significance > 0.05].sort_values('significance',ascending=False)

liguistic_columns = ['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb','auxverb','past','present','future','adverb','preps','conj','negate','quant','number']
liguistic_stats = stats.ix[liguistic_columns]
liguistic_stats[['significance','diff']][liguistic_stats.significance <= 0.05].sort_values('significance',ascending=False)

psychological_columns = ['swear','social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health','sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home','money','relig','death','assent','nonfl','filler']
psychological_stats = stats.ix[psychological_columns]
print(len(psychological_stats[psychological_stats.significance <= 0.05]))
psychological_stats[['significance','diff']][psychological_stats.significance <= 0.05].sort_values('significance',ascending=True)

## story categories
psychological_stats[psychological_stats.significance <= 0.05].sort_values('diff',ascending=False).head(10)

## non-story categories
psychological_stats[psychological_stats.significance <= 0.05].sort_values('diff',ascending=True).head(10)



