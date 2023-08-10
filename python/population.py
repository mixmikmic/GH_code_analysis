import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cluster=json.loads(file('../json/cluster.json','r').read())
citysave=json.loads(file('../json/citysave3.json','r').read())
N=json.loads(file('../json/N.json','r').read())

import wolframalpha
#app_id='T7449E-PXXTAHUHUA'
#nagyatom@yahoo.com
app_id='HHKXW4-Q6WJG2XAXW'
#csaladenespp@yahoo.com
client = wolframalpha.Client(app_id)

unicities={}
for i in cluster:
    if cluster[i] not in unicities:
        unicities[cluster[i]]=citysave[i]['country']

pop1=json.loads(file('../json/pop1c.json','r').read())
err1=json.loads(file('../json/pop1ec.json','r').read())
pop2=json.loads(file('../json/pop2c.json','r').read())
err2=json.loads(file('../json/pop2ec.json','r').read())
pop3=json.loads(file('../json/pop3c.json','r').read())
err3=json.loads(file('../json/pop3ec.json','r').read())
pop4=json.loads(file('../json/pop4c.json','r').read())
err4=json.loads(file('../json/pop4ec.json','r').read())

err=err1+err2+err3+err4
G={}
error=[]
len(err)

import unicodedata
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

for c in err:
    if c not in G.keys()+error:
        ys={"pop":0,"nearby":{}}
        q=remove_accents(strip_accents('population of '+c.split('/')[0].                                       replace('island','').                                       replace('Island','').strip()+', '+unicities[c]))
        res = client.query(q)
        good=True
        if 'pod' in res:
            for i in range(len(res['pod'])):
                try:
                    if res['pod'][i]['@title']=="Result":
                        x=res['pod'][i]['subpod']['plaintext']
                        if 'available' not in x:
                            popul=x[:x.find('people')-1]
                            if 'mill' in popul:
                                popul=popul[:popul.find('mill')-1]
                                if '|' in popul:popul=popul.split('|')[1].strip()
                            ys['pop']=int(float(popul)*1000000.0)
                            G[c]=ys
                            print 'partial success',c
                except: pass
                try:
                    if res['pod'][i]['@title']=="Nearby cities":
                        x=res['pod'][i]['subpod']['plaintext'].split('\n')
                        if 'available' not in x:
                            for y in x[:-1]:
                                people=y[y.rfind('|')+2:y.find('people')-1]
                                if 'mill' in people:
                                    people=float(people[:people.find('mill')-1])*1000000.0
                                km=float(y[y.find('|')+2:y.find(' km ')])
                                ys['nearby'][y.split('|')[0].split(',')[0].strip()]={"km":km,"people":int(people)}
                            G[c]=ys
                            print 'success',c
                            good=False
                except: pass
        if good: 
            print 'error',c
            error.append(c)

print len(G),len(error)

for i in pop1:
    if i in G:print i,1
    G[i]=pop1[i]
for i in pop2:
    if i in G:print i,2
    G[i]=pop2[i]
for i in pop3:
    if i in G:print i,3
    G[i]=pop3[i]
for i in pop4:
    if i in G:print i,4
    G[i]=pop4[i]

file("../json/pop_cities.json",'w').write(json.dumps(G))
file("../json/pope_cities.json",'w').write(json.dumps(error))
print len(G)

G={}
error=[]

for c in N:
    if c not in G.keys()+error:
        print c,
        q='population of '+c
        try:
            res = client.query(q)
            for i in range(len(res['pod'])):
                if res['pod'][i]['@title']=="Result":
                    x=res['pod'][i]['subpod']['plaintext']
                    popul=x[:x.find('people')-1]
                    if 'mill' in popul:
                            popul=float(popul[:popul.find('mill')-1])*1000000.0
                    G[c]=int(popul)
        except: error.append(c)

file("../json/pop_countries.json",'w').write(json.dumps(G))

error

for c in error:
    if c not in G.keys():
        print c,
        q='population of '+c
        try:
            res = client.query(q)
            for i in range(len(res['pod'])):
                if res['pod'][i]['@title']=="Result":
                    x=res['pod'][i]['subpod']['plaintext']
                    popul=x[:x.find('people')-1]
                    if 'mill' in popul:
                            popul=float(popul[:popul.find('mill')-1])*1000000.0
                    elif 'bill' in popul:
                            popul=float(popul[:popul.find('bill')-1])*1000000000.0
                    G[c]=int(popul)
        except: print c

cc={'FYR of Macedonia':'Macedonia',
u'São Tomé and Principe':'Sao Tome and Principe',
'Micronesia (Federated States of)':'Micronesia',
u"Lao People's Dem. Rep.":'Laos'}
for c in error:
    if c not in G.keys()+['Palestinian Territories']:
        print c,
        q='population of '+cc[c]
        res = client.query(q)
        for i in range(len(res['pod'])):
            if res['pod'][i]['@title']=="Result":
                x=res['pod'][i]['subpod']['plaintext']
                popul=x[:x.find('people')-1]
                if 'mill' in popul:
                        popul=float(popul[:popul.find('mill')-1])*1000000.0
                elif 'bill' in popul:
                        popul=float(popul[:popul.find('bill')-1])*1000000000.0
                G[c]=int(popul)

c='Palestinian Territories'
print c,
q='population of '+c
res = client.query(q)
for i in range(len(res['pod'])):
    s=0
    if res['pod'][i]['@title']=="Result":
        xx=res['pod'][i]['subpod']['plaintext'].split('\n')
        for x in xx[:-1]:
            popul=x[x.find('|')+2:x.find('people')-1]
            if 'mill' in popul:
                    popul=float(popul[:popul.find('mill')-1])*1000000.0
            elif 'bill' in popul:
                    popul=float(popul[:popul.find('bill')-1])*1000000000.0
            s+=int(popul)
        G[c]=s

file("../json/pop_countries2.json",'w').write(json.dumps(G))

