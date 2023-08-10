get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import mwapi
import mwtypes
import pandas
import seaborn

session = mwapi.Session("https://en.wikipedia.org", user_agent="ahalfaker@wikimedia.org -- IWSC demo")

WEIGHTS = {'Stub': 1, 'Start': 2, 'C': 3, 'B': 4, 'GA': 5, 'FA': 6}
def score2sum(score_doc):
    if score_doc is None:
        return None
    weighted_sum = 0
    for cl, proba in score_doc['probability'].items():
        weighted_sum += WEIGHTS[cl] * proba
    return weighted_sum

def fetch_wp10_score(rev_id):
    response = requests.get('https://ores.wikimedia.org/v3/scores/enwiki/{0}/wp10'.format(rev_id))
    try:
        return response.json()['enwiki']['scores'][str(rev_id)]['wp10']['score']
    except:
        return None


def fetch_wp10_scores(rev_ids):
    executor = ThreadPoolExecutor(max_workers=8)
    return executor.map(fetch_wp10_score, rev_ids)

def fetch_historical_scores(page_name):
    historical_scores = []
    for response_doc in session.get(action='query', prop='revisions', titles=page_name, 
                                    rvprop=['ids', 'timestamp','user'], rvlimit=100, rvdir="newer", 
                                    formatversion=2, continuation=True):
        rev_docs = response_doc['query']['pages'][0]['revisions']
        rev_ids = [d['revid'] for d in rev_docs]
        for rev_doc, score_doc in zip(rev_docs, fetch_wp10_scores(rev_ids)):
            rev_id = rev_doc['revid']
            user = rev_doc['user']
            timestamp = mwtypes.Timestamp(rev_doc['timestamp'])
            weighted_sum = score2sum(score_doc)
            historical_scores.append({'rev_id': rev_id, 'timestamp': timestamp, 'weighted_sum': weighted_sum,'user':user})
            sys.stderr.write(".")
            sys.stderr.flush()
        sys.stderr.write("\n")
    
    return historical_scores

def oresToPandas(title):
    historical_scores = pandas.DataFrame(fetch_historical_scores(title))
    historical_scores['time'] =pandas.to_datetime(historical_scores.timestamp, format='%Y-%m-%dT%H:%M:%SZ',errors='ignore')
    historical_scores = historical_scores.set_index('time')
    return historical_scores

data=oresToPandas('Ada Lovelace')

from peakdetect import peakdetect
from collections import Counter
def plotAndPeaks(df):
    indexes = peakdetect(df.weighted_sum, lookahead=1,delta=1)
    print('Max')
    for i in indexes[0]:
        print(df.index[i[0]],'https://en.wikipedia.org/w/?diff=prev&oldid=%s' % df.rev_id[i[0]],df.user[i[0]])
    print('Min')
    for i in indexes[1]:
        print(df.index[i[0]],'https://en.wikipedia.org/w/?diff=prev&oldid=%s' % df.rev_id[i[0]],df.user[i[0]],df.weighted_sum[i[0]])
    return df['weighted_sum'].plot()

def vandalCandidates(df):
    vandals = []
    indexes = peakdetect(df.weighted_sum, lookahead=1, delta=1)
    for i in indexes[1]:
        vandals.append(df.user[i[0]])
        #print(df.index[i[0]],'https://en.wikipedia.org/w/?diff=prev&oldid=%s' % df.rev_id[i[0]],df.user[i[0]])
    candidates = [x for x,y in Counter(vandals).most_common(5)]
    for i in indexes[1]:
        if df.user[i[0]] in candidates:
            #print(df.index[i[0]-1],'https://en.wikipedia.org/w/?diff=prev&oldid=%s' % df.rev_id[i[0]],df.user[i[0]])
            pass
    print(candidates)
    return None
 
    
    

plotAndPeaks(data)

vandalCandidates(data)



