import csv
import sys
import os
import glob
import pandas as pd
import nltk
import matplotlib.pylab as plt
from IPython.display import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def isNotNull(value):
    return value is not None and len(value)>0

neg_file = []
os.chdir('C:\Users\Miya\OneDrive\Miya\'sGithub\Text-Sentiment-Analysis-/neg')
for file in glob.glob('*.txt'):
    neg_file.append(file)
neg_content = []
for i in range(len(neg_file)):
    txt = open(neg_file[i])
    neg_content.append(txt.read())

pos_file = []
os.chdir('C:\Users\Miya\OneDrive\Miya\'sGithub\Text-Sentiment-Analysis-/pos')
for file in glob.glob('*.txt'):
    pos_file.append(file)
pos_content = []
for i in range(len(pos_file)):
    txt = open(pos_file[i])
    pos_content.append(txt.read())

Bing_senti = pd.DataFrame()
Bing_senti['pos'] = pos_content
Bing_senti['neg'] = neg_content
Bing_senti.head() 
Bing_senti.head()

os.chdir('C:\Users\Miya\OneDrive\Miya\'sGithub\Text-Sentiment-Analysis-/')
dict_pos = []
dict_neg = []
f = open('negative-words.txt','r')
for line in f:
    t= line.strip().lower();
    if (isNotNull(t)):
        dict_neg.append(t)
f.close()

f = open('positive-words.txt','r')
for line in f:
    t = line.strip().lower();
    if (isNotNull(t)):
        dict_pos.append(t)
f.close()

analysis_for_pos = []
for i in range(len(Bing_senti)):
    tokens = nltk.word_tokenize(pos_content[i])
    neg_cnt = 0
    pos_cnt = 0
    for neg in dict_neg:
        if (neg in tokens):
            neg_cnt = neg_cnt +1
    for pos in dict_pos:
        if (pos in tokens):
            pos_cnt = pos_cnt +1
    analysis_for_pos.append(pos_cnt - neg_cnt)     
Bing_senti['Bing_analysis_for_pos'] = analysis_for_pos

analysis_for_neg = []
for i in range(len(Bing_senti)):
    tokens = nltk.word_tokenize(neg_content[i])
    neg_cnt = 0
    pos_cnt = 0
    for neg in dict_neg:
        if (neg in tokens):
            neg_cnt = neg_cnt +1
    for pos in dict_pos:
        if (pos in tokens):
            pos_cnt = pos_cnt +1
    analysis_for_neg.append(pos_cnt - neg_cnt)     
Bing_senti['Bing_analysis_for_neg'] = analysis_for_neg

Bing_senti.head()

Bing_senti.Bing_analysis_for_pos.hist(bins = 20,color = 'pink')
plt.title('Sentiment Analysis for Positive Reviews Distribution',fontsize = 20)
plt.savefig('figure_1.png')
Image('figure_1.png')

Bing_senti.Bing_analysis_for_neg.hist(color = 'pink')
plt.title('Sentiment Analysis for Negative Reviews Distribution',fontsize = 20)
plt.savefig('figure_2.png')
Image('figure_2.png')

neg_analysis_label = []
for i in analysis_for_neg:
    if i >0:
        neg_analysis_label.append(1)
    else:
        neg_analysis_label.append(0)
        
pos_analysis_label = []
for i in analysis_for_pos:
    if i >0:
        pos_analysis_label.append(1)
    else:
        pos_analysis_label.append(0)

Bing_senti['analysis_label_for_neg'] = neg_analysis_label
Bing_senti['analysis_label_for_pos'] = pos_analysis_label

Bing_senti['label_for_neg'] = [0]*len(Bing_senti)
Bing_senti['label_for_pos'] = [1]*len(Bing_senti)
Bing_senti.head()

Bing_analysis = Bing_senti.analysis_label_for_neg.tolist() + Bing_senti.analysis_label_for_pos.tolist()

True_label = Bing_senti.label_for_neg.tolist() + Bing_senti.label_for_pos.tolist()

confusion_matrix(True_label,Bing_analysis)

print classification_report(True_label,Bing_analysis)

Bing_senti.to_csv('Bing_Output.csv')

