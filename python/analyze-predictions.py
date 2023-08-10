import pandas as pd

data = pd.read_csv('new_glasses.csv').iloc[:,:-1] # Last column is a bug

elems = data.columns[:-4]

data['composition'] = data[elems].apply(lambda x: ''.join('%s%d'%(e,x*100) for e,x in x.items() if x > 0), axis=1)

data['system'] = data[elems].apply(lambda x: '-'.join(sorted(e for e,x in x.items() if x > 0)), axis=1)

data.head(5)

top_alloys = data.sort_values('P(gfa=AM)_predicted', ascending=False).drop_duplicates('system', 'first')

top_alloys.head(8)[['composition', 'system', 'P(gfa=AM)_predicted']]

top_alloys.query('Hf == 0').head(8)[['composition', 'system', 'P(gfa=AM)_predicted']]



