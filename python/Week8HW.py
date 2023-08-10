import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df= pd.read_csv("../data/SMSSpamCollection",sep='\t', names=['spam', 'txt'])

df.head()



