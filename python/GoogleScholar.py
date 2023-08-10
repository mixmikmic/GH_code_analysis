import scholar as gss
import pandas as pd

import time
import numpy as np

querier = gss.ScholarQuerier()
settings = gss.ScholarSettings()

querier.apply_settings(settings)

phrase='quantitative "susceptibility mapping" mri'
query = gss.SearchScholarQuery()
query.set_phrase(phrase)

num_total=1780
num_steps=89

for ii in range(43, num_steps):
    ind_shift=20*ii
    query.start_with=ind_shift
    querier.send_query(query)
    for i in range(0,len(querier.articles)):
        temp=querier.articles[i].as_csv(header=True).split('\n')[1].split('|')
        df.loc[i+1+ind_shift]=temp[:11]  #ind starts with 1
    time_pause=np.random.randint(10, 30)
    print('Step {} is done! {} articles downloaded. Sleep {} seconds.'.format(ii,ii*20+20,time_pause))
    time.sleep(time_pause)

df2=df.iloc[0:972,:]

df2.to_csv('CitationData2.csv', index=False)

df.to_pickle('CitationData.pickle')

