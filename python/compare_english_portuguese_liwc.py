import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

portuguese_liwc = pd.read_csv('portuguese_stories_liwc.csv.gz', compression='gzip')
english_liwc = pd.read_csv('icwsm09_stories_liwc.csv.gz', compression='gzip')

print("postuguese: " + str(len(portuguese_liwc)))
print("english: " + str(len(english_liwc)))

portuguese_liwc = portuguese_liwc[((portuguese_liwc['wc'] > 10) & (portuguese_liwc['wc'] < 1000))]
portuguese_liwc = portuguese_liwc[((portuguese_liwc['wps'] > 3) & (portuguese_liwc['wps'] < 30))]
portuguese_liwc = portuguese_liwc[portuguese_liwc['i'] > 2]
portuguese_liwc = portuguese_liwc[(portuguese_liwc['negemo'] + portuguese_liwc['posemo']) > 2]
portuguese_liwc = portuguese_liwc[portuguese_liwc['score'] > 0]
len(portuguese_liwc)

english_liwc = english_liwc[((english_liwc['wc'] > 10) & (english_liwc['wc'] < 1000))]
english_liwc = english_liwc[((english_liwc['wps'] > 3) & (english_liwc['wps'] < 30))]
english_liwc = english_liwc[english_liwc['i'] > 2]
english_liwc = english_liwc[(english_liwc['negemo'] + english_liwc['posemo']) > 2]
english_liwc = english_liwc[english_liwc['score'] > 0]
len(english_liwc)

english_sample = english_liwc.sample(len(portuguese_liwc))

def diff(x): 
    return abs(x['portuguese_mean'] - x['english_mean'])

stats = pd.DataFrame(data={'portuguese_mean': portuguese_liwc.mean(axis=0)}, index=portuguese_liwc.columns.values)
stats['english_mean'] = english_sample.mean(axis=0)

stats['diff'] = stats.apply(diff,axis=1)

significance = []
for column in list(stats.index.values):
    a = portuguese_liwc[column]
    b = english_sample[column]
    t, p = wilcoxon(a, b)
    significance.append(p)
stats['significance'] = significance

stats[stats.significance > 0.05].sort_values('significance',ascending=False)

stats[stats.significance < 0.05].sort_values('diff',ascending=True).head(10)

stats[stats.significance < 0.05].sort_values('diff',ascending=False).head(10)

