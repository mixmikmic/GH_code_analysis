import pandas as pd
import re
import time
import csv
from time import sleep
import numpy as np

pop = pd.read_csv('https://raw.githubusercontent.com/jamesthomson/Evolution_of_Pop_Lyrics/master/data/scraped_lyrics.tsv',sep='\t')

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"<br />", " ", string) #Replace HTML break with white space
	string = re.sub(r"br", " ", string)
	string = re.sub(r"\\", " ", string)
	return string.strip().lower()

pop_clean = pop[pop['lyrics']!='Lyrics Not found']

x_text = [clean_str(sent) for sent in pop_clean.lyrics]



def replace_with_oov(input_str,vocab):
    result=''
    for word in input_str.split():
        if (word in vocab):
            result= result + word + ' '
        else:
            result= result + '<oov> '
    return result


word_count = {} # Keys are words, Values are frequency

for review in x_text:

    words = review.split()

    for word in words:
        try:
            word_count[word]+=1
        except:
            word_count[word]=0


res = list(sorted(word_count, key=word_count.__getitem__, reverse=True))

global vocab
vocab = res[:10000]

# Replacing words that are not in the vocab with '<oov>'
cleaned_x_text = [replace_with_oov(item,vocab) for item in x_text]

def get_tagged_lyric(str_input):
    tagged_lyric = (str_input).replace('\r\n\r\n','</l></s><s><l>')
    tagged_lyric = (tagged_lyric).replace('\r\n','</l><l>')
    return '<s><l>'+tagged_lyric+'</l></s>'

pattern_1 = '\([0-9]+x\)'
pattern_2 = '\[.*?\]'
pattern_3 = '\{.*?\}'
pattern_4 = 'chorus'
pattern_5 = 'verse'

all_patterns = [pattern_1,pattern_2,pattern_3,pattern_4,pattern_5]

final_lyrics = []

for lyric in cleaned_x_text:

    try:
        lyric = lyric.lower()
        for pattern in all_patterns:
            lyric = re.sub(pattern,'',lyric)
            
        final_lyrics.append(get_tagged_lyric(lyric))
    except:
        print "There was a problem"

pop_clean['Final_lyrics']=final_lyrics

# Before cleaning
print pop_clean.lyrics[1000]

# After cleaning
print pop_clean.Final_lyrics[1000]

pop_clean.to_csv('data/pop_clean_lyrics_dataset.csv')



