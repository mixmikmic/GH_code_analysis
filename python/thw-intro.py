import pandas as pd
df = pd.read_csv('WorkshopSurvey.csv')
topics = df.filter(regex=r'^Please.', axis=1)
topics.columns = [t.split('[')[1][:-1] for t in topics.columns]
print('Number of responses: {}'.format(len(topics)))
topics.head(3)

interest = topics.apply(pd.Series.value_counts)
interest = interest.reindex(['Very interested', 'Interested',
                             'Only slightly interested', 'Not interested'])

get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
figsize(10, 8)
interest.T.sort_values('Very interested').plot(kind='bar', stacked=True)

for row in df['Suggest other topics for Workshops, Tutorials or Seminars'].dropna():
    print(row)
    print('\n')

