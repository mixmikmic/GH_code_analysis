import os
import shutil
import string

os.chdir('/home/harish/PycharmProjects/Topic-Modeling/Data Extraction/dataset/')

# #reading each file names
# for fn in os.listdir('.'):
#      if os.path.isfile(fn):
#             tx = open(fn, 'r+')
#             print(fn)
            

import nltk
from nltk.corpus import stopwords
from nltk import SnowballStemmer

stopwords = nltk.corpus.stopwords.words('english')

custom_stopwords = ["system","reserve","tthe","rnthe",
                    "participants", "continue", "open","committee",
                    "federal", "also", "meeting", "members", 
                    "FOMC", "\r","\t","Present", "\n", 'year',"discussion", 'turned','authority', 'member','members','manager','january', 'february', 'march', 
                    'april', 'may', 'june', 'july','august', 'september', 'october', 
                    'november', 'december',"system","reserve","rate", "continue", 
                    "open","committee", "federal", "market",
                    "recent", "meeting", "FOMC", "\r","\t","Present", "\n", 'year']

for word in custom_stopwords:
    stopwords.append(word)

path = '/home/harish/PycharmProjects/Topic-Modeling/Data Extraction/dataset/FullDataset/'

for root, dirs, files in os.walk(path):
    for fn in files:
        lst = []
        os.chdir(root)
        fin = open(fn, "r")
#         directory = root+"/out/"
        directory = path + 'yearbasis/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        

        for w in fin.read().lower().split():
            each_word = w.lower().strip(string.punctuation)#March, becomes march
            if each_word not in stopwords:
                lst.append(w)
        
        filename = str(fn) 
        if len(filename) > 16:
            fname = filename[11:15]
        else:
            fname = filename[:4]
        
        fout = open(directory+ fname, "a")
        fout.write(" ".join(lst))
        
        fin.close()
        fout.close()

st = 'fomcminutes20080805.txt'
st[11:15]

# #reading each file names
# path = '/home/harish/PycharmProjects/Topic-Modeling/Data Extraction/dataset/'
# output = '/home/harish/PycharmProjects/Topic-Modeling/Data Extraction/dataset/1993_2017/'
# for root, dirs, files in os.walk(path):
#     for fn in files:
#         lst = []
#         os.chdir(root)
#         fin = open(fn, "r")
# #         directory = root+"/out/"
# #         directory = path + 'pdftotext/'
#         directory = output
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        
#         fout = open(directory+ fn, "w+")
        
#         Flag = False
#         before = None
#         count = 0
#         for w in fin.read().lower().split():
#             each_word = w.lower().strip(string.punctuation)#March, becomes march
#             if before == 'committee' and each_word ==  'ratified':
#                 Flag = True
#                 count +=1
#                 print(count)
#             if each_word not in stopwords and Flag == True:
#                 lst.append(w)
# #                 if  each_word == 'march,':
# #                     print(w)
#             before = each_word
            
        
#         fout = open(directory+ fn, "w+")
#         fout.write(" ".join(lst))
        
#         fin.close()
#         fout.close()

#reading each file names
path = '/home/harish/PycharmProjects/Topic-Modeling/Data Extraction/dataset/'
for root, dirs, files in os.walk(path):
    for fn in files:
        lst = []
        os.chdir(root)
        fin = open(fn, "r")
        directory = root+"/out/"
        
        try:
            shutil.rmtree(directory)
        except:
            pass


s = "committee\r\nratified"
s.replace("\r"," ").replace("\n"," ").replace("\t"," ")
# print(strin[0])
# if strin in "committee ratified":
#     print("s")
s

num = "Sdfdsf\1232sdff"
num.replace("\1"," ")



