import numpy as np
import spacy
import os
from collections import Counter
import torch
import glob
from spacy import attrs

batch_size = 1000
prev_count = 184680

nlp = spacy.load('en') # loads default English object
cnn_dir = '/home/irteam/users/data/CNN_DailyMail/dailymail/stories/'
cnn_tok_dir = '/home/irteam/users/data/CNN_DailyMail/dailymail/1.stories_tokenized/'

file_list = [os.path.join(cnn_dir,file) for file in os.listdir(cnn_dir)]
total_files = len(file_list)
files_read = 0
count = 0

for file in file_list[prev_count:]:
    with open(file) as f:
        text = f.read()
#         text = text.lower()
#         text = text.replace('\n\n',' ')
#         text = text.replace('(cnn)','')
#         text = text.split("@highlight")
#         body = text[0]
    body_words = [x.text for x in list(nlp(text))]
    out = ' '.join(body_words)
#     print(out)
    out_file = file.replace(cnn_dir,cnn_tok_dir)
    with open(out_file,'w') as f:
        f.write(out)
    count+=1
    if count%100==0:
        print(count)

count



