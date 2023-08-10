import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')

def gen():
    with open('AMiner-Paper.txt', 'r',  encoding="utf8") as f:
        datum={}
        citations =0
        row=0
        readFile = f.readlines()
        for line in readFile:
            
            if '#index' in line:
                if bool(datum):
                    datum['citations'] = citations
                    try:
                        for i in range(len(datum['author'])):
                            
                            datum_to_save = datum.copy()
                            datum_to_save['author']=datum['author'][i]
                            datum_to_save['affiliation']=datum['affiliation'][i]
                            yield datum_to_save
                            row+=1
                    except IndexError as e:
                        continue
                    
                    datum={}
                    citations =0
                datum['id'] = line[7:].rstrip()
                
            elif '#*' in line:
                datum['title'] = line[3:].rstrip()
            elif '#@' in line:
                datum['author'] = line[3:].rstrip().rsplit(";")
            elif '#o' in line:
                datum['affiliation'] = line[3:].rstrip().rsplit(";")
            elif '#t' in line:
                datum['year'] = line[3:].rstrip()
            elif '#c' in line:
                datum['venue'] = line[3:].rstrip()
            elif '#%' in line:
                citations +=1
            elif '#!' in line:
                datum['abstract'] = line[3:].rstrip()


data = pd.DataFrame(gen(), columns =('id', 'title', 'author', 'affiliation', 'year', 
                                  'venue', 'citations', 'abstract'))
data['year'] =pd.to_numeric(data['year'], errors = 'coerce').fillna(0)

data.describe()

plt.scatter(data["year"], data["citations"])

data = data[data["citations"] >0]
data = data[data["year"] >0]

plt.scatter(data["year"], data["citations"], facecolors='none', edgecolors='blue')
plt.axvline(x=2001, color ='red', linewidth=2.0)

plt.hist(data[data["year"]==2002]['citations'])

plt.boxplot(data[data["year"]==2002]['citations'])

data['treatment'] = data['year'] >= 2001
data.drop_duplicates(subset='id')

lm = smf.ols("citations ~ year + treatment + treatment*year", data).fit()
lm.params

lm.summary()

