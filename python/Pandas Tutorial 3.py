import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt  
pd.set_option('max_columns',50)  
get_ipython().magic('matplotlib inline')

# pass in column names for each csv  
u_cols = ['user_id','age','sex','occupation','zip_code']  
users = pd.read_csv('./data/ml-100K/u.user',sep='|',names=u_cols,encoding='latin-1')  

r_cols = ['user_id','movie_id','rating','unix_timestamp']  
ratings = pd.read_csv('./data/ml-100k/u.data',sep='\t',names=r_cols,encoding='latin-1')  

# the movies file contains columns indicating the movie's genres  
# let's only load the first five columns of the file with usecols  

m_cols = ['movie_id','title','release_date','video_release_date','imdb_url']   
movies = pd.read_csv('./data/ml-100k/u.item',sep='|',names=m_cols,usecols=range(5),encoding ='latin-1')

# create one merged Data Frame  
movie_ratings = pd.merge(movies,ratings)  
lens = pd.merge(movie_ratings,users)

movies.head(2)

ratings.head(2)

movie_ratings.head(2)

users.head(2)

lens.head(2)

most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]  
most_rated

lens.title.value_counts()[:25]

movie_stats = lens.groupby('title').agg({'rating':[np.size,np.mean]})  
movie_stats.head()  

# sort by rating average.  
movie_stats.sort_values([('rating','mean')], ascending = False).head()

atleast_100 = movie_stats['rating']['size'] >=100  

movie_stats[atleast_100].sort_values([('rating','mean')],ascending=False)[:15]

most_50 = lens.groupby('movie_id').size().sort_values(ascending=False)[:50] 

users.age.plot.hist(bins=30)  
plt.title("Distribution of users' ages")  
plt.ylabel('count of users')  
plt.xlabel('age') ;

labels = ['0-9' , '10-19' , '20-29' ,'30-39','40-49','50-59','60-69','70-79']  
lens['age_group'] = pd.cut(lens.age,range(0,81,10),right=False,labels=labels)  

lens.head()

lens[['age','age_group']].drop_duplicates()[:10] 

lens.groupby('age_group').agg({'rating':[np.size,np.mean,np.median]})

lens.set_index('movie_id',inplace=True)  

most_50.head()

by_age=lens.loc[most_50.index].groupby(['title','age_group'])
by_age.agg({'rating':[np.mean]}).head(15)  

#same as  
#by_age=lens.loc[most_50.index].groupby(['title','age_group']).agg({'rating':[np.mean]}) 
#by_age.head(15)    

by_age.rating.mean().unstack(1).fillna(0)[10:20]

by_age.rating.mean().unstack(0).fillna(0)[10:20]

lens.reset_index('movie_id',inplace=True)  

lens.head(2)

pivoted = lens.pivot_table(index =['movie_id','title'],
                           columns=['sex'],values='rating',fill_value=0)  
pivoted.head()

pivoted['diff'] = pivoted.M - pivoted.F  
pivoted.head()  

pivoted.reset_index('movie_id',inplace=True)  

disagreements = pivoted[pivoted.movie_id.isin(most_50.index)]['diff']
disagreements.sort_values().plot(kind='barh',figsize=[9,15])  
plt.title('Male Vs Female Avg. Ratings\n(Difference >0 = Favored by Men)') 
plt.ylabel('Title')  
plt.xlabel('Average Rating Difference') ;



