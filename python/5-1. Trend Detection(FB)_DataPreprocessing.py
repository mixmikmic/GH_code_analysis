import pandas as pd
disease = pd.read_csv('Final_FB_2804_utf16.csv', encoding='UTF-16LE')
display(disease.head(3))

df = pd.DataFrame(disease, columns = ['key','id.x','created_time.x', 'message.x','created_time.y', 'message.y'])
rm_duplicates_x = df.drop_duplicates(subset=['key','message.x'])
rm_duplicates_y = df.drop_duplicates(subset=['key','message.y'])
#print(len(df),len(rm_duplicates_x),len(rm_duplicates_y))
#display(rm_duplicates_x.head(10))
#display(rm_duplicates_y.head(10))
message_x=rm_duplicates_x[['key', 'id.x', 'created_time.x', 'message.x']]
message_y=rm_duplicates_y[['key', 'id.x', 'created_time.y', 'message.y']]
message_x.columns=['key', 'id.x', 'created_time','message']
message_y.columns=['key', 'id.x', 'created_time','message']
#display(message_x.head(10))
#display(message_y.head(10))
#print(len(message_x),len(message_y))
frames = [message_x, message_y]
result = pd.concat(frames, keys=['message.x', 'message.y'])
display(result.head(3))
print(len(df),len(result))
result.to_csv("post_comment_utf16.csv", encoding='UTF-16LE',columns = ['key','id.x','created_time', 'message'])

post_comment = pd.read_csv('post_comment_utf16.csv', encoding='UTF-16LE') 
#df.apply(lambda x: pd.lib.infer_dtype(x.values))
post_comment.columns=['xy','docid','key', 'id.x', 'created_time','message']
display(post_comment.head(3))
display(len(post_comment['key']))

rm_na = post_comment.dropna()
dtime = rm_na.sort(['created_time'])
dfinal = pd.DataFrame(dtime, columns = ['xy','docid','key','id.x', 'created_time','message'])
dfinal.to_csv("post_comment_rm_utf16.csv", encoding='UTF-16LE',columns = ['xy','docid','key','id.x', 'created_time','message'])

dstart = pd.read_csv('post_comment_rm_utf16.csv', encoding='UTF-16LE')
display(dstart.head(3))
print(len(dstart['key']))

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
def _calculate_languages_ratios(text):
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios
def detect_language(text):
    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language

import pandas as pd
import numpy as np
import re
import nltk
import csv
nltk.download('stopwords')
from nltk.corpus import stopwords
#pip install langdetect
from langdetect import detect

df_fb = pd.read_csv('post_comment_rm_utf16.csv', encoding = 'utf-16LE', sep=',')
                         
with open('post_comment_preprocessing_utf16.csv', 'w', encoding='UTF-16LE', newline='') as csvfile:
    column = [['no','xy','docid','key','id.x','created_time','message','lang','re_message']]
    writer = csv.writer(csvfile)
    writer.writerows(column)
    
for i in range(len(df_fb['message'])):    
    features = []
    features.append(i)
    features.append(df_fb['xy'][i])
    features.append(df_fb['docid'][i])
    features.append(df_fb['key'][i])
    features.append(df_fb['id.x'][i])
    features.append(df_fb['created_time'][i])
    features.append(df_fb['message'][i])
    if(str(df_fb['message'][i])=="nan"):
        features.append('english')
        features.append(df_fb['message'][i])
    else:
        #tokens = nltk.word_tokenize(str(df_twitter['message'][i]))
        tokens=' '.join(re.findall(r"[\w']+", str(df_fb['message'][i]))).lower().split()
        postag=nltk.pos_tag(tokens)
        irlist=[',','.',':','#',';','CD','WRB','RB','PRP','...',')','(','-','``','@']
        wordlist=['co', 'https', 'http','rt','www','ve','don',"i'm","it's"]
        adjandn = [word for word,pos in postag if pos not in irlist and word not in wordlist and len(word)>2]
        #adjandn = [word for word,pos in postag if pos not in irlist]
        lang=detect_language(df_fb['message'][i])
        features.append(lang)
        #print(i,lang)
        stop = set(stopwords.words(lang))
        wordlist = [i for i in adjandn if i not in stop]
        features.append(' '.join(wordlist))
    with open('post_comment_preprocessing_utf16.csv', 'a', encoding='UTF-16LE', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])

df_postncomment = pd.read_csv('post_comment_preprocessing_utf16.csv', encoding = 'UTF-16LE', sep=',')
#display(df_postncomment.head(5))
df_english= df_postncomment.loc[df_postncomment['lang'] == 'english']
df_rm = df_english.drop_duplicates(subset=['key','re_message'])
display(df_rm.head(3))
print(len(df_postncomment['key']),len(df_rm['key']))
rm_english_na = df_rm.dropna()
#print(len(rm_english_na))
dfinal_fb = pd.DataFrame(rm_english_na, columns = ['no','xy','docid','key','id.x','created_time','message','lang','re_message'])
dfinal_fb.to_csv("final_preprocessing_utf16.csv", encoding='UTF-16LE',columns = ['no','xy','docid','key','id.x','created_time','message','lang','re_message'])

df_postn = pd.read_csv('final_preprocessing_utf16.csv', encoding = 'UTF-16LE', sep=',')
display(df_postn.head(3))
print(len(df_postn))



