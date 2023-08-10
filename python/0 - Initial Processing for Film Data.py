import numpy as np
import pandas as pd
import json
import pickle
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Create an empty dataframe and define its columns based on my data.
film_df = pd.DataFrame(columns=['actors','budget','countries','directors','genres','gross_usa','keywords',
                                  'languages', 'mpaa', 'rating', 'release_date', 'runtime',
                                   'title', 'votes', 'writers', 'year'])

#Import all my json files to create one big dataframe.
for i in range(1, 51):
    
    file = "../Desktop/film_data_files_json/film_data_" + str(i) + ".json"

    with open(file, "r") as f:
        data = f.read()
        
    temp_df = pd.DataFrame(json.loads(data))
    
    film_df = film_df.append(temp_df)

len(film_df)

film_df.head()

#I won't use these columns because the data is pretty uninformative, so I will delete them.
del film_df['keywords']
del film_df['year']
del film_df['title']
del film_df['release_date']

#And drop null values.
film_df = film_df.dropna(how='any', axis=0)
film_df.reset_index(drop=True, inplace=True)

len(film_df)

#Budget was imported as strings, so I am going to process it a bit and check the results.
film_df['budget'] = film_df['budget'].apply(lambda x: x.strip())
film_df = film_df[film_df['budget'].apply(lambda x: re.split(r'(\d+)', x)[0]  == '$')]
film_df.reset_index(drop=True, inplace=True)
film_df['budget'] = film_df['budget'].apply(lambda x: int(x.split('$')[1].replace(',', '')))
film_df['budget'].describe()

#Plotting the distribution
plt.figure(figsize=(15,5))
y = film_df['budget']/1000000
y.hist(bins = 50)
plt.xlabel('Millions of dollars')
plt.ylabel('Count');

#Do the same for USA Domestic Gross
film_df['gross_usa'] = film_df['gross_usa'].apply(lambda x: x.strip())
film_df = film_df[film_df['gross_usa'].apply(lambda x: re.split(r'(\d+)', x)[0]  == '$')]
film_df.reset_index(drop=True, inplace=True)
film_df['gross_usa'] = film_df['gross_usa'].apply(lambda x: int(x.split('$')[1].replace(',', '')))
film_df['gross_usa'].describe()

#There is a zero value there, so we will remove this row
film_df.loc[film_df['gross_usa'] == 0].index
film_df = film_df.drop(film_df.index[3588])
film_df['gross_usa'].describe()

#Plotting the distribution
plt.figure(figsize=(15,5))
y = film_df['gross_usa']/1000000
y.hist(bins = 50)
plt.xlabel('Millions of dollars')
plt.ylabel('Count');

#Process votes, since they were imported as strings and need to be turned into integeres.
film_df['votes'] = film_df['votes'].str.replace(',','').apply(lambda x: int(x))
film_df['votes'].describe()

#Plotting the distribution
plt.figure(figsize=(15,5))
y = film_df['votes']
y.hist(bins = 100)
plt.xlabel('Number of votes')
plt.ylabel('Count');

#Check rating
film_df['rating'] = film_df['rating'].str.replace(',','').apply(lambda x: float(x))
film_df['rating'].describe()

#Looks good, so I will plot its distribution.
plt.figure(figsize=(8,5))
y = film_df['rating']
y.hist(bins = 10)
plt.xlabel('Rating')
plt.ylabel('Counts');

#Check if there are any empty strings for genres
film_df.loc[film_df['genres'] == ''].index

#No, so now I will fix genres.
film_df['genres'] = film_df['genres'].apply(lambda x: str(x).split('+_+')[1:])

#Check mpaa
film_df.loc[film_df['mpaa'] == ''].index

#Check languages
film_df.loc[film_df['languages'] == ''].index

#Fix languages
film_df['languages'] = film_df['languages'].apply(lambda x: x.strip().split('+_+')[1:])

#Check directors
film_df.loc[film_df['directors'] == ''].index

#Delete the empty line
film_df = film_df.drop(index = 3604).reset_index(drop=True)

#Check dataframe length
len(film_df)

#Format directors and pick only the first one
film_df['directors'] = film_df['directors'].apply(lambda x: x.split('+_+')[1].replace(',', '')) 

#Look at the data to check if it looks ok.
film_df['directors'].value_counts().head()

film_df['directors'].value_counts().tail()

#Check writers
film_df.loc[film_df['writers'] == ''].index

#Drop the empty row
film_df = film_df.drop(index = 658).reset_index(drop=True)

#Process strings and pick the first value.
film_df['writers'] = film_df['writers'].apply(lambda x: x.split('+_+')[1].replace(',', '')) 

film_df['writers'].value_counts().head()

#Check for unique values.
film_df['writers'].nunique()

#Get top-3 actors
film_df['actors'] = film_df['actors'].apply(lambda x: x.strip().split('+_+')[1:4])

#Work on countries.
film_df.loc[film_df['countries'] == ''].index

film_df['countries'] = film_df['countries'].apply(lambda x: x.strip().split('+_+')[1:])

#Check the dataframe now.
film_df.head()

len(film_df)

#Store the data
pickle_out = open("processed_film_data","wb")
pickle.dump(film_df, pickle_out)
pickle_out.close()

pickle_in = open("processed_film_data","rb")
test_film_df = pickle.load(pickle_in)

test_film_df.head()

