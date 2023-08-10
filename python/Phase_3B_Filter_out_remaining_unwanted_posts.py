java_frameworks_list = []
with open('./data/java_frameworks.txt') as file:
    for line in file:
        line = line.replace(' ', '')
        java_frameworks_list.append(line.lower().strip())

program_lang_list = []
with open('./data/programming_languages_list.txt') as file:
    for line in file:
        line = line.replace('-', '').replace(' ', '').replace('/', '').replace('(', '').replace(')', '').strip()
        program_lang_list.append(line.lower())

import pandas as pd
df = pd.read_csv('./data/filtered_data_ready_for_app.csv')

df.shape

count = 0
for index, row in df.iterrows():
    found = False
    #extract tags from each row
    post_tags = df.loc[index, 'Tags'].split()
    #check if the tags are present in stop lists (frameworks, other languages)
    for tag in post_tags:
        tag = tag.replace('-', '')
        #if an unwanted tag found - remove from the dataframe
        if tag in program_lang_list or tag in java_frameworks_list:
            found = True
    if found:
        df.drop(index, inplace=True)
        #df.drop(df.index[i], inplace=True)

df.to_csv('./data/filtered_cleaned_posts_no_frameworks_no_alt_lang.csv', index=False)

#size of resulting dataset
df.shape



