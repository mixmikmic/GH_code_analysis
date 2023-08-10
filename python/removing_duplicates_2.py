data = pd.read_pickle('pickels/17k_apperal_data')

# This takes significant amount of time...
# O(n^2) time.
# Takes about an hour to run on a decent computer

indices=[]
for i,row in data.iterrows():
        indices.append(i)
        
stage2_dedupe_asins=[]

while len(indices)!=0:
    i=indices.pop()
    stage2_dedupe_asins.append(data['asin'].loc[i])
    a=data['title'].loc[i].split()
    
    for j in indices:
        b=data['title'].loc[j].split()
        length=max(len(a),len(b))
        
        count=0
        
        for k in itertools.zip_longest(a,b):
            if (k[0]==k[1]):
                count+=1
        
        if (length-count) <3:
            indices.remove(j)

data=data.loc[data['asin'].isin(stage2_dedupe_asins)]

print('Number of data points after stage 2 of dedupe : ', data.shape[0])
#from 17k apparels we reduced to 16k apparels 

data.pickle('pickels/16k_apperal_data')

data=pd.read_pickle('pickels/16k_apperal_data')





