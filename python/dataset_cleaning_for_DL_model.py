import pandas

df1 = pandas.read_csv('./dataset/selected_song_based_on_tags.csv', sep='\t')
df2 = pandas.read_csv('./dataset/selected_song_based_on_tags (copy).csv', sep='\t')

print df1.columns
print df2.columns

print 'songs with preview files: ' + str(len(df1[df1['preview_file']!='not found']))
print 'songs with mood data: ' + str(len(df2[df2['moods']!='not found']))

minimized_df1 = df1.drop(['title', 'tags', 'genres', 'preview_url'], axis=1)
print (minimized_df1.columns)
minimized_df1.columns = ['track_id', 'song_id', 'title', 'preview_file']
print (minimized_df1.columns)

minimized_df2 = df2.drop(['tags', 'song_id', 'title', 'preview_url', 'preview_file', 'preview_info', 'gracenote'], axis=1)
print (minimized_df2.columns)

# use_df = pandas.merge(minimized_df1, minimized_df2, on='track_id', how='inner')
# minimized_df2.head()

def clean_cell(content):
    char_to_delete = ['[', ']', "'", '"']

    for i in char_to_delete:
#         print i
        content=content.replace(i, '')
    return content

use_df = pandas.merge(minimized_df1, minimized_df2, on='track_id', how='inner')
print(len(use_df))
use_df = use_df[(use_df['title'] != 'not found') & (use_df['preview_file'] != 'not found')  & (use_df['genres'] != 'not found') & (use_df['moods'] != 'not found') & (use_df['tempos'] != 'not found')]
print(len(use_df))
a = use_df['preview_file'][0]
print(a[24:])

def fixPreviewFile(content):
    return content[24:]

use_df['preview_file'] = use_df['preview_file'].apply(fixPreviewFile)

use_df.head()

cleaned_df = use_df
cleaned_df['genres'] = cleaned_df['genres'].apply(clean_cell)
cleaned_df['moods'] = cleaned_df['moods'].apply(clean_cell)
cleaned_df['tempos'] = cleaned_df['tempos'].apply(clean_cell)

cleaned_df.head()

genres_df = pandas.DataFrame()

moods_df = pandas.DataFrame()

tempos_df = pandas.DataFrame()

def df_splitter(orig_df, axis, label_concern):
    temp_df = pandas.DataFrame()
    for idx, row in orig_df.iterrows():
        items = row[label_concern].split(',')
        for i in items:
            temp_df = temp_df.append([[row[axis], i.strip()]])
            
    return temp_df

genres_df = df_splitter(cleaned_df, 'track_id', 'genres')
moods_df = df_splitter(cleaned_df, 'track_id', 'moods')
tempos_df = df_splitter(cleaned_df, 'track_id', 'tempos')

genres_df.columns = ['track_id', 'genre']  
grouped_genres = genres_df.groupby('genre').agg({'genre': 'count'}).rename(columns={'genre': 'genre_count'}).sort_values('genre_count', ascending=False)
print(len(grouped_genres))
print grouped_genres.index
# for i in grouped_genres.index:
#     print i
selected_genre = grouped_genres.head(43)
print(selected_genre.index)
selected_genre_df = genres_df[genres_df['genre'].isin(list(selected_genre.index))].reset_index(drop=True)
print len(genres_df)
print len(selected_genre_df)
print len(selected_genre_df['track_id'].unique())
selected_genre.head(43)

moods_df.columns = ['track_id', 'mood']
grouped_moods = moods_df.groupby('mood').agg({'mood': 'count'}).sort_values('mood', ascending=False)
print(len(grouped_moods))
print grouped_moods.index
# for i in grouped_moods.index:
#     print i

happiness_file = open('./dataset/deep_learning/mood_class/happiness.txt', 'rU')
anger_file = open('./dataset/deep_learning/mood_class/anger.txt', 'rU')
sadness_file = open('./dataset/deep_learning/mood_class/sadness.txt', 'rU')
neutral_file = open('./dataset/deep_learning/mood_class/neutral.txt', 'rU')

def iterateContent(orig_file):
    temp = []
    for line in orig_file.readlines():
        temp.append(line.strip())
    return temp

selected_moods_df = moods_df.copy()

happiness = iterateContent(happiness_file)
anger = iterateContent(anger_file)
sadness = iterateContent(sadness_file)
neutral = iterateContent(neutral_file)

selected_moods_df.loc[selected_moods_df['mood'].isin(happiness), 'mood'] = 'happiness'
selected_moods_df.loc[selected_moods_df['mood'].isin(anger), 'mood'] = 'anger'
selected_moods_df.loc[selected_moods_df['mood'].isin(sadness), 'mood'] = 'sadness'
selected_moods_df.loc[selected_moods_df['mood'].isin(neutral), 'mood'] = 'neutral'
selected_moods_df.reset_index(drop=True, inplace=True)

print selected_moods_df.groupby('mood').agg({'mood': 'count'}).rename(columns={'mood': 'mood_count'}).sort_values('mood_count', ascending=False).head()
moods_df.groupby('mood').agg({'mood': 'count'}).rename(columns={'mood': 'mood_count'}).sort_values('mood_count', ascending=False).head()

tempos_df.columns = ['track_id', 'tempo']
grouped_tempos = tempos_df.groupby('tempo').agg({'tempo': 'count'}).sort_values('tempo', ascending=False)
print(len(grouped_tempos))
print(grouped_tempos.index)
# for i in grouped_tempos.index:
#     print i

slow_file = open('./dataset/deep_learning/tempo_class/slow.txt', 'rU')
medium_file = open('./dataset/deep_learning/tempo_class/medium.txt', 'rU')
fast_file = open('./dataset/deep_learning/tempo_class/fast.txt', 'rU')

def iterateContent(orig_file):
    temp = []
    for line in orig_file.readlines():
        temp.append(line.strip())
    return temp

selected_tempos_df = tempos_df.copy()

slow = iterateContent(slow_file)
medium = iterateContent(medium_file)
fast = iterateContent(fast_file)

selected_tempos_df.loc[selected_tempos_df['tempo'].isin(slow), 'tempo'] = 'slow'
selected_tempos_df.loc[selected_tempos_df['tempo'].isin(medium), 'tempo'] = 'medium'
selected_tempos_df.loc[selected_tempos_df['tempo'].isin(fast), 'tempo'] = 'fast'

print len(selected_tempos_df)
print len(selected_tempos_df['track_id'].unique())
print len(selected_tempos_df.drop_duplicates())

selected_tempos_df.reset_index(drop=True, inplace=True)

# grouped_tempos
selected_tempos_df.groupby('tempo').agg({'tempo': 'count'}).rename(columns={'tempo': 'tempo_count'}).sort_values('tempo_count', ascending=False).head()

song_data_df = use_df.drop(['genres', 'moods', 'tempos'], axis=1)
genre_label_map_df = pandas.DataFrame(columns = ['track_id'] + list(selected_genre.index))

for idx,row in selected_genre_df.iterrows():
#     print genre_label_map_df[genre_label_map_df['track_id'] == row['track_id']].empty
    a = row['genre']
    if genre_label_map_df[genre_label_map_df['track_id'] == row['track_id']].empty:
        genre_label_map_df = genre_label_map_df.append({
            'track_id': row['track_id'],
            row['genre']: 1
        }, ignore_index=True).fillna(0)
    else:
#         print idx
#         print row['track_id']
#         print genre_label_map_df.head()
#         print genre_label_map_df.iloc[0]
#         genre_label_map_df.iloc[genre_label_map_df['track_id'] == row['track_id']]+=1
#         print genre_label_map_df.loc[genre_label_map_df['track_id'] == row['track_id']][row['genre']]
        genre_label_map_df.loc[genre_label_map_df['track_id'] == row['track_id'], [row['genre']]]=1
#         print genre_label_map_df.loc[genre_label_map_df['track_id'] == row['track_id']][row['genre']]
#         break

genre_label_map_df.head()

# display = True

# def labelCounter1(series):
#     count = 0
#     global display
#     if display == True:
#         display=False
#         print series
#     for i in series:
#         if i == 'Urban':
#             count+=1
#     return count

# def labelCounter2(series):
#     count = 0
#     for i in series:
#         if i == 'Rock':
#             count+=1
#     return count

# genre_agg = {}

# # for genre in list(selected_genre.index):
# for genre in ['Urban','Rock']:
#     print genre
#     def labelCounter(series):
#         count = 0
#         global display
#         if display == True:
#             print series
#             print genre
#         for i in series:
#             if i == genre:
#                 if display == True:
#                     print '+1'
#                 count+=1
#         display=False
#         return count
#     genre_agg[genre] = labelCounter

# use_genre_songs_df = selected_genre_df.groupby('track_id').agg({
# #     'genre': {
# #         'Urban': labelCounter1,
# #         'Rock': labelCounter2
# #         }
#     'genre': genre_agg
#     })
    
# print len(use_genre_songs_df[use_genre_songs_df[('genre', 'Urban')] == 1])
# use_genre_songs_df.sort_values([('genre', 'Urban')], ascending=False).head()
# # use_genre_songs_df.iloc[630]
# # use_genre_songs_df.iloc[use_genre_songs_df.index.get_level_values('track_id') == 'TRAAAEF128F4273421']
# # use_genre_songs_df.columns
# # selected_genre_df[selected_genre_df['genre'] == 'Alternative & Punk']

# import inspect
# selected_genre_df[selected_genre_df['track_id']=='TRAAAEF128F4273421']
# selected_genre_df[selected_genre_df['genre']=='Urban']
# selected_genre_df.iloc[0]

# use_genre_songs_df.iloc[227]
# use_genre_songs_df.iloc[use_genre_songs_df.index.get_level_values('track_id') == 'TRAAAAW128F429D538']

unnormalized_mood_label_map_df = pandas.DataFrame(columns = ['track_id'] + ['happiness', 'sadness', 'anger', 'neutral'])

for idx,row in selected_moods_df.iterrows():
#     print unnormalized_mood_label_map_df[unnormalized_mood_label_map_df['track_id'] == row['track_id']].empty
    a = row['mood']
    if unnormalized_mood_label_map_df[unnormalized_mood_label_map_df['track_id'] == row['track_id']].empty:
        unnormalized_mood_label_map_df = unnormalized_mood_label_map_df.append({
            'track_id': row['track_id'],
            row['mood']: 1
        }, ignore_index=True).fillna(0)
    else:
        unnormalized_mood_label_map_df.loc[unnormalized_mood_label_map_df['track_id'] == row['track_id'], [row['mood']]]=1
        unnormalized_mood_label_map_df.loc[unnormalized_mood_label_map_df['track_id'] == row['track_id'], [row['mood']]]+=1
        

unnormalized_mood_label_map_df.head()

selected_moods_df[selected_moods_df['track_id'] == 'TRAABLR128F423B7E3']

mood_label_map_df = unnormalized_mood_label_map_df
# for idx,row in mood_label_map_df.iterrows():
# #     print row
#     total_mood = row['happiness'] + row['sadness'] + row['anger'] + row['neutral']
# #     print total_mood
# #     print mood_label_map_df.happiness.iloc[idx]
# #     print mood_label_map_df.happiness.iloc[idx]
#     mood_label_map_df.happiness.iloc[idx] /= total_mood
#     mood_label_map_df.sadness.iloc[idx] /= total_mood
#     mood_label_map_df.anger.iloc[idx] /= total_mood
#     mood_label_map_df.neutral.iloc[idx] /= total_mood
# #     break
# mood_label_map_df.head()

tempo_label_map_df = pandas.DataFrame(columns = ['track_id'] + ['slow', 'medium', 'fast'])

# for idx,row in selected_tempos_df.iterrows():
# #     print tempo_label_map_df[tempo_label_map_df['track_id'] == row['track_id']].empty
#     a = row['tempo']
#     if tempo_label_map_df[tempo_label_map_df['track_id'] == row['track_id']].empty:
#         tempo_label_map_df = tempo_label_map_df.append({
#             'track_id': row['track_id'],
#             row['tempo']: 1
#         }, ignore_index=True).fillna(0)
#     else:
#         tempo_label_map_df.loc[tempo_label_map_df['track_id'] == row['track_id'], [row['tempo']]]=1

# tempo_label_map_df.head()

print len(mood_label_map_df[mood_label_map_df['']==0])
mood_label_map_df = mood_label_map_df[mood_label_map_df['']==0]
print len(mood_label_map_df[mood_label_map_df['Other']==0])
mood_label_map_df = mood_label_map_df[mood_label_map_df['Other']==0]
print len(mood_label_map_df[mood_label_map_df['Other']==0])

print len(tempo_label_map_df[tempo_label_map_df['']==0])
tempo_label_map_df = tempo_label_map_df[tempo_label_map_df['']==0]
print len(tempo_label_map_df[tempo_label_map_df['']==0])

# saving the new synthesized datasets
label_map_df = pandas.merge(song_data_df, genre_label_map_df, on='track_id', how='inner')
print label_map_df.columns
mood_label_map_df = mood_label_map_df.drop(['Other', ''], axis=1)
label_map_df = pandas.merge(label_map_df, mood_label_map_df, on='track_id', how='inner')
print label_map_df.columns
tempo_label_map_df = tempo_label_map_df.drop([''], axis=1)
label_map_df = pandas.merge(label_map_df, tempo_label_map_df, on='track_id', how='inner')
print label_map_df.columns
label_map_df.reset_index(inplace=True, drop=True)
print label_map_df.columns

print len(label_map_df)
label_map_df.head()

# label_map_df.to_csv('./dataset/deep_learning/label_map(fraction_mood).csv', sep='\t', encoding='utf-8', index=False)
# label_map_df.to_csv('./dataset/deep_learning/label_map(boolean_mood).csv', sep='\t', encoding='utf-8', index=False)

use_df.to_csv('./dataset/deep_learning/combined_data.csv', sep='\t', encoding='utf-8', index=False)
song_data_df.to_csv('./dataset/deep_learning/song_data.csv', sep='\t', encoding='utf-8', index=False)
genres_df.to_csv('./dataset/deep_learning/genre_song_pair.csv', sep='\t', encoding='utf-8', index=False)
moods_df.to_csv('./dataset/deep_learning/mood_song_pair.csv', sep='\t', encoding='utf-8', index=False)
tempos_df.to_csv('./dataset/deep_learning/tempo_song_data.csv', sep='\t', encoding='utf-8', index=False)

