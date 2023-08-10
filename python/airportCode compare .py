import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')


df = pd.read_csv('db_21/1987.csv', delimiter=',')
print len(df['Origin'])




import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')


df = pd.read_csv('db_21/1987.csv', delimiter=',')
print len(df['Origin'])

uniqueAirport=[]            
for temp in df['Origin']:
    if temp not in uniqueAirport:
        uniqueAirport.append(temp)

print len(uniqueAirport) 

print uniqueAirport[:10]


print len(df['Dest'])

uniqueAirportDest=[]               
for temp in df['Dest']:
    if temp not in uniqueAirportDest:
        uniqueAirportDest.append(temp)

print len(uniqueAirportDest) 

df91 = pd.read_csv('db_21/1991.csv', delimiter=',')
print len(df91['Origin'])

uniqueAirportDest91=[]               
for temp in df['Origin']:
    if temp not in uniqueAirportDest91:
        uniqueAirportDest91.append(temp)

print len(uniqueAirportDest91) 

#codes is a the file containing - [airport codes , lat , lon]

dfAll = pd.read_csv('codes.csv', delimiter=',')
print len(dfAll['locationID'])


x = dfAll['locationID']; 

uniqueAirportCompare=[]                
for temp in x:
    if temp in uniqueAirportDest91:
        uniqueAirportCompare.append(temp)

print len(uniqueAirportCompare) 
print uniqueAirportCompare[:10]

#test is just trivial compared and can be ignored 
dft = pd.read_csv('test.csv', delimiter=',')
print len(dft['DEST'])

uniqueAirporttest=[]            
for temp in dft['DEST']:
    if temp not in uniqueAirport:
        uniqueAirporttest.append(temp)

print uniqueAirporttest[:10]

uniqueAirportCompare=[]                
for temp in x:
    if temp in uniqueAirportDest91:
        uniqueAirportCompare.append(uniqueAirporttest)

print len(uniqueAirportCompare) 
print uniqueAirportCompare[:1]










