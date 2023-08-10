import pandas as pd
from datetime import datetime, timedelta,date

def convertToDateString(date):
    return (datetime(1970, 1, 1) + timedelta(milliseconds=date)).strftime("%Y-%m-%d %H:%M:%S")

file_path="medium_search_dataScience/outputs_scrapped/April232018/"
post_file="Posts_20180423_224224.csv"
posts=pd.read_csv(file_path+post_file,encoding='utf-8')
print "Shape of Posts Data is"+ str(posts.shape)

posts['createdDatetime']=posts['createdAt'].apply(convertToDateString)
posts['firstPublishedDatetime']=posts['firstPublishedAt'].apply(convertToDateString)
posts['latestPublishedDatetime']=posts['latestPublishedAt'].apply(convertToDateString)
posts['updatedDatetime']=posts['updatedAt'].apply(convertToDateString)
#posts[['createdAt','createdDatetime']]
posts.to_csv(file_path+"Posts_Preprocessed.csv",encoding='utf-8',index=False)


cols_to_consider=['id','creatorId','firstPublishedDatetime','createdDatetime','ScrappingDate','collectionId','isSubscriptionLocked'
                  ,'language','linksCount','readingTime','recommends','responsesCreatedCount','subTitle','tags_name','title','totalClapCount','wordCount','imageCount']

posts=posts[posts.inResponseToPostId.isnull()]
posts.shape

posts=posts[cols_to_consider]

posts.shape

posts.to_csv(file_path+"Posts_Extracted.csv",encoding='utf-8')

def convertStringToDate(dateString,formatString='%Y-%m-%d %H:%M:%S'):
    dates=datetime.strptime(dateString,formatString)
    return dates.date()

def getDaysBetweenDates(date1,date2):
    delta=date1-date2
    return delta.days
    

posts['firstPublishedDate']=posts['firstPublishedDatetime'].apply(convertStringToDate,args=('%Y-%m-%d %H:%M:%S',))
posts['createdDate']=posts['createdDatetime'].apply(convertStringToDate,args=('%Y-%m-%d %H:%M:%S',))

posts['ScrappedDate']=posts['ScrappingDate'].apply(convertStringToDate,args=('%d-%m-%Y',))

posts["PublishAndScrapping_Difference_Days"]=(posts["ScrappedDate"]-posts["firstPublishedDate"]).dt.days

features=['totalClapCount',
          'PublishAndScrapping_Difference_Days',
          'isSubscriptionLocked',
          'responsesCreatedCount',
         'recommends'
         ]

posts_cluster=posts[features]
posts_cluster.shape

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

def getCluster(data,num_cluster=2):
    kmeans=KMeans(n_clusters=num_cluster)
    kmeans.fit(data)
    return [kmeans.labels_,kmeans.cluster_centers_]

#We want to divide data into popular and non-popular



posts['two_cluster_label']=getCluster(posts_cluster,num_cluster=2)[0]
posts['three_cluster_label']=getCluster(posts_cluster,num_cluster=3)[0]

posts['two_cluster_label'].value_counts()
#Having two cluster gave us 35 articles in one cluster and 765 articles in another

posts['three_cluster_label'].value_counts()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
colors = {0:'red', 1:'blue',2:'green',3:'orange'}

def plotScatter(x,y,hues,xlabel,ylabel):
    plt.scatter(x,y,c=hues.apply(lambda x: colors[x]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plotScatter(posts['totalClapCount'],posts['recommends'],posts['two_cluster_label'],"#Claps","#Recommends")

plotScatter(posts['totalClapCount'],posts['responsesCreatedCount'],posts['two_cluster_label'],"#Claps","#Responses")

plotScatter(posts['totalClapCount'],posts['PublishAndScrapping_Difference_Days'],posts['two_cluster_label'],"#Claps","Days Between Publishing and Scrapping")

#Let us build clusters after standardising - commented as this didnt give good clusters
'''
from sklearn.preprocessing import StandardScaler
posts_cluster_standardised = StandardScaler().fit_transform(posts_cluster)
posts['two_cluster_label_after_scaling']=getCluster(posts_cluster_standardised,num_cluster=2)[0]
posts['two_cluster_label_after_scaling'].value_counts()

'''

plotScatter(posts['totalClapCount'],posts['recommends'],posts['three_cluster_label'],"#Claps","#Recommends")

plotScatter(posts['totalClapCount'],posts['responsesCreatedCount'],posts['three_cluster_label'],"#Claps","#Responses")

plotScatter(posts['totalClapCount'],posts['PublishAndScrapping_Difference_Days'],posts['three_cluster_label'],"#Claps","Days Between Publishing and Scrapping")

#Rename the clusters and write to csv
clusters={0:'less popularity',1:'high popularity',2:'medium popularity'}
posts['two_cluster_popularity']=posts['two_cluster_label'].apply(lambda x: clusters[x])
posts['three_cluster_popularity']=posts['three_cluster_label'].apply(lambda x:clusters[x])
posts.to_csv(file_path+"Posts_with_Popularity.csv",encoding='utf-8',index=False)

# Let us create a few features
import re
def getWordCount(text):
    return len(re.findall(r'\w+', text))

posts['TitleWordCount']=posts['title'].apply(getWordCount)

from wordcloud import WordCloud
'''
Create corpus of Titles - join all titles togther
'''
titles=posts['title']
text=""
for title in titles:
    text = text+" "+"".join(title).encode('utf-8').strip()
text
text=text.lower()
wordcloud = WordCloud(relative_scaling=0.5,max_words=300).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=(40,40))
plt.show()


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

tags=posts['tags_name']

from nltk.tokenize import wordpunct_tokenize

tags=[tag.replace("[","") for tag in tags]
tags=[tag.replace("]","") for tag in tags]

tags_str=""
for tag in tags:
    tags_str = tags_str+","+"".join(tag).encode('utf-8').strip()

tags_str=tags_str.strip(",")
words=tags_str.split(",")
words=[x for x in words if x]
words=[x.strip() for x in words]
from collections import Counter
counts = Counter(words)
#words


tags_count=pd.DataFrame.from_dict(counts,orient="index")

tags_count=tags_count.reset_index()
#tags_count.columns=['Tags','Frequency']
tags_count=tags_count.rename(columns={'index':'Tags', 0:'Frequency'})

#Number of Tags 
tags_count['Tags'].nunique()
#740 unique Tags were present

tags_count.to_csv(file_path+'TagsCount_DataScience.csv',index=False)

posts[["tags_name","id"]].to_csv("Tags_Name.csv",encoding='utf-8')

Counter(words).most_common(10)

from collections import Counter
def getTagsCount(tags):
    tags=[tag.replace("[","") for tag in tags]
    tags=[tag.replace("]","") for tag in tags]
    tags_str=""
    for tag in tags:
        tags_str = tags_str+","+"".join(tag).encode('utf-8').strip()
    tags_str=tags_str.strip(",")
    words=tags_str.split(",")
    words=[x for x in words if x]
    words=[x.strip() for x in words]
 
    counts = Counter(words)
    tags_count=pd.DataFrame.from_dict(counts,orient="index")
    tags_count=tags_count.reset_index()

    tags_count=tags_count.rename(columns={'index':'Tags', 0:'Frequency'})
    return tags_count
    


posts=pd.read_csv(file_path+"Posts_with_Popularity.csv")

high_popular_posts=posts[posts['three_cluster_popularity']=='high popularity']

high_popular_posts_tags=high_popular_posts['tags_name']
tags_count_high_popular=getTagsCount(high_popular_posts_tags)

tags_count_high_popular
tags_count_high_popular.to_csv(file_path+"High_Popular_Tags.csv",index=False)



