import pandas as pd

df_posts=pd.read_pickle('posts_2017-07-29 12_50_03_789969.p')

df_posts

df_comments=pd.read_pickle('comments_2017-07-29 12_50_03_789969.p')

df_comments

df_posts['react_haha'].nlargest(10)

df_posts['react_haha'].nlargest(50)

df_posts.ix[df_posts['react_sad'].nlargest(1).index]['description']

# Comments count 
gr=df_comments.groupby(['from_name']).count()

# Top commenters
gr.sort_values('from_id', ascending=False)

df_com = df_comments.drop(['created_time'], axis=1)



mf=pd.merge(df_com, df_posts, on='post_id', how='inner')
mf

# Dataframe from particular commenter
mf[mf['from_name']=='Anu Talu']

text_df=df_posts[['message', 'description']]

# Word count of news headers
text_df['description'].str.split(' ', expand=True).stack().value_counts()

#sona Eesti arv on suurem kui ei sona:)



import estnltk

# Ordered posts with reactions sad
sad = df_posts.sort_values(['react_sad'], ascending=[False]).nlargest(40, columns='react_sad')
sad

# Ordered posts with reactions haha
haha = df_posts.nlargest(40, columns = 'react_haha')
haha

# Words in posts with sad reactions
sad_w=sad['description'].str.split(' ', expand=True).stack().value_counts()
sad_w

# Words in posts with haha reactions
haha_w=haha['description'].str.split(' ', expand=True).stack().value_counts()
haha_w

# Transform to word count data
sad_w.columns = ['word', 'count']
sad_w.to_csv('sad_word.csv')
haha_w.columns = ['word', 'count']
haha_w.to_csv('haha_word.csv')

#Estonian language Natural Language Processing ToolKit
#from estnltk.taggers.adjective_phrase_tagger.adj_phrase_tagger import AdjectivePhraseTagger
#from estnltk import Text

















s_dict

posts_from = mf[mf['from_name']=='Anu Talu']
posts_from

import os
count = 0
with open('./output_short.txt', 'a') as f1:
    score = {}
    for index, row in mf.iterrows(): #Data frame for iteration
        message = row ['message_x']
        sub_score = 0
        #if count<600000:
            #f1.write(message + os.linesep)   
        
        try:
            for word in message.split(' '):
                sub_score = sub_score + int(s_dict[word]) if word in s_dict else 0
            score[index] = sub_score 
            #print (sub_score, message)
        except AttributeError:
            pass





