import pandas as pd

df = pd.read_excel(open('c:\data\multi-label-data.xlsx','rb'))

#def clean_words(data_):
    
#    return data_

data_ = df["Detailed Description"].str.replace('.*@+.*', ' ')
data_ = data_.str.replace('\t',' ')
data_ = data_.str.replace(',',' ')
data_ = data_.str.replace(':',' ')
data_ = data_.str.replace(';',' ')
data_ = data_.str.replace('=',' ')
data_ = data_.str.replace('\x08','\\b') # \b is being treated as backspace
data_ = data_.str.replace('_',' ')
data_ = data_.str.replace('(',' ')
data_ = data_.str.replace(')',' ')
data_ = data_.str.replace('+',' ')
data_ = data_.str.replace('-',' ')
data_ = data_.str.replace('`',' ')
data_ = data_.str.replace('\'',' ')
data_ = data_.str.replace('.',' ')
data_ = data_.str.replace('//',' ')
data_ = data_.str.replace('\\',' ')
data_ = data_.str.replace('/',' ')
data_ = data_.str.replace('_',' ')
data_ = data_.str.replace('"',' ')
data_ = data_.str.replace('\r\n', ' ')
data_ = data_.str.replace('\n', ' ')
data_ = data_.str.replace('*', ' ')
data_ = data_.str.replace('-', ' ')
data_ = data_.str.replace('#', ' ')
data_ = data_.str.replace('\s+', ' ')
data_ = data_.str.replace('=', ' ')
data_ = data_.str.replace('_', ' ')
data_ = data_.str.replace('>', ' ')
data_ = data_.str.replace('\n\t\r', ' ')
data_ = data_.str.replace('  ', ' ')
data_ = data_.str.replace(',', ' ')
data_



data_1 = df["Categorization"].str.replace('.*@+.*', ' ')
data_1 = data_1.str.replace(':', ' ')
data_1 = data_1.str.replace(',', ' ')

data_2 = df["Configuration item"].str.replace('.*@+.*', ' ')
data_2 = data_2.str.replace(':', ' ')
data_2 = data_2.str.replace(',', ' ')

df["Detailed Description"] = data_
df["Categorization"] = data_1
df["Configuration item"] = data_2

df.columns = ['category', 'config_item', 'description']

##xx= df.category.unique().tolist()
##yy = df.config_item.unique()
##xx

#df["description"] = df["Detailed Description"]
#df = df.drop(["Detailed Description"], axis=1)

#df["category"] = df["Categorization"]
#df = df.drop(["Categorization"], axis=1)

#df["config_item"] = df["Configuration item"]
#df = df.drop(["Configuration item"], axis=1)
#df
#df.to_csv("c:\\data\\full_dataset_clean.csv", sep=',', encoding='utf-8')

#type(df)
#df.values.tolist()
x = df["description"].values.tolist()
y = df.drop(["description"], axis = 1) #remaining 2 columns are in here
y = y.values.tolist()
y



# converting all categories in to column and marking it as 1 in the matrix

from sklearn.preprocessing import MultiLabelBinarizer

# convert labels to binary forms (similar to pivot) 
mlb = MultiLabelBinarizer()
y_enc = mlb.fit_transform(y)



df_plot = pd.DataFrame(y_enc) 
df_plot["description"] = x
df_plot

counts = []
cats = list(df_plot.columns.values)
cats.remove("description")
#cats
for i in cats:
  counts.append((i, df_plot[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', '#count'])
#df_stats

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

df_stats.plot(x='category', y='#count', kind='bar', legend=False, grid=True, figsize=(15, 8))

from sklearn.cross_validation import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y_enc, test_size=0.33)

# introduce k-fold cross validation for better utilization of the data

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english') + list(punctuation))

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)

#from sklearn.feature_extraction.text import CountVectorizer
#vect = CountVectorizer()

# train_x_vct = vect.fit_transform(train_x)
# test_x_vct = vect.transform(test_x)

train_x_vct = vectorizer.fit_transform(train_x)
test_x_vct = vectorizer.transform(test_x)

# from sklearn.svm import SVC
# clf = OneVsRestClassifier(SVC(probability=True))
# clf.fit(train_x_vct, train_y)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

classifier = OneVsRestClassifier(LinearSVC(random_state=42))
# classifier = OneVsRestClassifier(LogisticRegression(random_state=42))
# classifier = OneVsRestClassifier(RandomForestClassifier())
classifier.fit(train_x_vct, train_y)



predictions = classifier.predict(test_x_vct)
predictions

test_y

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

precision = precision_score(test_y, predictions, average='micro')
recall = recall_score(test_y, predictions, average='micro')
f1 = f1_score(test_y, predictions, average="micro") 
 
print("Micro-average quality numbers")
print("Precision Score: ", precision)
print("Recall Score: ", recall)
print("f1-score: ", f1)

# precision not to be considered
#print("Accuracy Score: ",accuracy_score(test_y, predictions))

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),('clf', OneVsRestClassifier(LinearSVC()))])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}

from sklearn.model_selection import GridSearchCV

grid_search_tune = GridSearchCV(classifier, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(train_x, train_y)

print("Best parameters set:")
print()
print(grid_search_tune.best_estimator_.steps)
print()

# measuring performance on test set

best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(test_x)

#print(predictions)
#print(test_y)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
#precision = precision_score(test_y, predictions)
#recall = recall_score(test_y, predictions)
print("Micro-average quality numbers")
print("-----------------------------")
print("f1-score : ",f1_score(test_y, predictions, average="micro"))
print("precision: ",precision_score(test_y, predictions, average="micro"))
print("recall   : ",recall_score(test_y, predictions, average="micro"))
print()
print("Micro-average quality numbers")
print("-----------------------------")
print("f1-score : ",f1_score(test_y, predictions, average="macro"))
print("precision: ",precision_score(test_y, predictions, average="macro"))
print("recall   : ",recall_score(test_y, predictions, average="macro"))





