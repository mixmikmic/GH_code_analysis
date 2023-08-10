import pandas as pd
df = pd.read_csv('./data/StackoverflowCompleteDS_JAVA.csv')

df = df[['Id', 'Title','Body','Tags', 'body']]

df.head()

def clean_tags(raw_tags):
    cleaned_tags = raw_tags.replace('>', " ").replace('<', " ").replace('java', '')
    return cleaned_tags

for index, row in df.iterrows():
    cleaned_tags = clean_tags(df.loc[index, 'Tags'])
    df.loc[index, 'Tags'] = cleaned_tags

df['title_tag_chunk'] = df[df.columns[1:3]].apply(lambda x: ','.join(x),axis=1)

import pickle
trained_NB_model = pickle.load(open('./models/multinomialnb_classifier_ngrams_title_tag.sav', 'rb'))

import numpy as np

tmp_df = df[50:100]

test_list = list(tmp_df['title_tag_chunk'])

len(trained_NB_model.predict(test_list))

for l in test_list:
    prediction = trained_NB_model.predict([l])
    #print("Question: {}\n Prediciton: {}\n".format(l, prediction))

#create the target column
df['OK'] = None

#iterates over the dataframe
for index, row in df.iterrows():
    
    #extract the correct data to feed model
    data = df.loc[index, 'title_tag_chunk']
    #predicts whether or not it is ok
    prediction = trained_NB_model.predict([data])
    #saves prediction to row
    df.loc[index, 'OK'] = prediction

df.OK.count()

#df

df['OK'] = df['OK'].str.get(0)

df_ok = df[df.OK == 1]

df_ok = df_ok.drop(['title_tag_chunk', 'OK'], axis=1)

test_list = list(df_ok.Tags.head(10))

#df_ok

df_ok.to_csv('./data/filtered_data_ready_for_app.csv', index=False)

'''
for item in list(df_ok.Title):
    print(item)
'''







