import pandas as pd, json

c=pd.read_excel('../../universal/countries/cnc.xlsx').columns

d=[]
for i in c:
    d.append({'country':i,'c':i[0],"countries":1,"type":"countries"})
d.append({'country':'Total','c':'Total',"countries":0,"type":"countries"})

file('c.json','w').write(json.dumps(d))

df=pd.read_html('https://en.wikipedia.org/wiki/List_of_sovereign_states')

h=[]
for i in df[0].index[4:225]:
    k=df[0].loc[i][0]
    if 'ZZZ' not in k:
        if u'→' not in k:
            if 'Gambia' not in k:
                if u'–' in k:k=k[:k.find(u'–')]
                if '[' in k:k=k[:k.find('[')]
                h.append({'country':k,'c':k[0],"countries":1,"type":"countries"})
            else: 
                h.append({'country':'Gambia','c':'G',"countries":1,"type":"countries"})
h.append({'country':'Total','c':'Total',"countries":0,"type":"countries"})

file('h.json','w').write(json.dumps(h))

