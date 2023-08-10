data=pd.read_pickle('pickels/28k_apparel_data')

data.head()

#Removing data with short titles as they do not give any detailed relevant description
data_sorted=data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short decription: ",data_sorted.shape[0])

# sort the whole data in alphabetical order of title
data_sorted.sort_values('title',inplace=True, ascending=False)
data_sorted.head()

print(data_sorted.shape[0])
print(data_sorted.shape[1])

indices=[]
for i,row in data_sorted.iterrows():
    indices.append(i)

import itertools
stage1_dedupe_asins=[]
i=0
j=0
num_data_points=data_sorted.shape[0]
while i<num_data_points and j<num_data_points:
    
    previous_i=i
    
    #store the list of words of ith title in a
    a=data['title'].loc[indices[i]].split()
    
    #search for similar title sequentially
    j=i+1
    while j<num_data_points:
        #store the words of jth title in b
        b=data['title'].loc[indices[j]].split()
        
        #store the max length of two strings
        length=max(len(a),len(b))
        
        #count stores the no of words that are present in both titles
        count=0
        
        #counting the no of same words in a,b
        for k in itertools.zip_longest(a,b):
            if (k[0]==k[1]):
                count+=1
        
        if (length-count)>2:
            stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])
            
            if j==num_data_points-1:
                stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[j]])
            
            i=j
            break
        else:
            j+=1
    if previous_i==i:
        break
        

data=data.loc[data['asin'].isin(stage1_dedupe_asins)]

print('Number of data points : ',data.shape[0])

data.to_pickle('pickels/17k_apperal_data')

