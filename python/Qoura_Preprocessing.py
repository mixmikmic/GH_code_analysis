import pandas as pd
import numpy as np
df = pd.read_csv("df_fe_without_preprocessing_train.csv")
df.fillna('')

import re
import pandas as pd
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


SAFE_DIV = 0.0001 
# To get the results in 4 decemal points

STOP_WORDS = stopwords.words("english")

#print STOP_WORDS


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    #print "Removing Punctuations..."
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    #print "Removing HTML tags and performing stemming..."
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    

#Preprocess the Questions:
df["question1"] = df["question1"].fillna("").apply(preprocess)
df["question2"] = df["question2"].fillna("").apply(preprocess)

df.head()

def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence of Que. in the Tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features



# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    
    # Merging Features with dataset
    
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
   
    #Computing Fuzzy Features and Merging with Dataset
    print("fuzzy features..")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df


df.columns

print("Extracting features for train:")
train_df = pd.read_csv("train.csv")
train_df = extract_features(train_df)
#train_df.drop(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"], axis=1, inplace=True)

# Creating new .csv File with total 21 features:
train_df.to_csv("nlp_features_train.csv", index=False)




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


import os
import gc
dfp = pd.read_csv("nlp_features_train.csv")
dfp.fillna('')
dfp.head()

import numpy as np
dfp_d = dfp[dfp['is_duplicate'] == 1]
dfp_nd = dfp[dfp['is_duplicate'] == 0]

# Converting 2d array of q1 and q2 and flatten the array: like {{1,2},{3,4}} to {1,2,3,4}
p = np.dstack([dfp_d["question1"], dfp_d["question2"]]).flatten()

n=np.dstack([dfp_nd["question1"], dfp_nd["question2"]]).flatten()

print len(p)
print len(n)

#Saving the np array into a text file
np.savetxt('train_p.txt', p, delimiter=' ', fmt='%s')
np.savetxt('train_n.txt', n, delimiter=' ', fmt='%s')

# Import the Required lib packages for WORD-Cloud generation
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# reading the text files and removing the Stop Words:
d = path.dirname('/home/abhishek/Documents/')

textp_w = open(path.join(d, 'train_p.txt')).read()
textn_w = open(path.join(d, 'train_n.txt')).read()
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("br")
stopwords.add(" ")
stopwords.remove("not")

stopwords.remove("no")
#stopwords.remove("good")
#stopwords.remove("love")
stopwords.remove("like")
#stopwords.remove("best")
#stopwords.remove("!")
type(textp_w)
print len(textp_w)

wc = WordCloud(background_color="white", max_words=len(textp_w),
stopwords=stopwords)
# generate word cloud
wc.generate(textp_w)

print "Word Cloud for Duplicate Question pairs"
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

wc = WordCloud(background_color="white", max_words=len(textn_w),
stopwords=stopwords)
# generate word cloud
wc.generate(textn_w)

print "Word Cloud for non-Duplicate Question pairs:"
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

n = dfp.shape[0]
sns.pairplot(dfp[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:n], hue='is_duplicate')

print "Scatter plot of token_sort_ratio & csc_min:"

sns.FacetGrid(dfp, hue="is_duplicate", size=9)    .map(plt.scatter, "token_sort_ratio", "csc_min")    .add_legend()

print "Scatter plot of token_sort_ratio & cwc_min:"
sns.FacetGrid(dfp, hue="is_duplicate", size=9)    .map(plt.scatter, "token_sort_ratio", "cwc_min")    .add_legend()

print "Scatter plot of token_sort_ratio & ctc_min:"
sns.FacetGrid(dfp, hue="is_duplicate", size=9)    .map(plt.scatter, "token_sort_ratio", "ctc_min")    .add_legend()

# Distribution of the token_sort_ratio
plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'token_sort_ratio', data = dfp[0:] , )

plt.subplot(1,2,2)


sns.distplot(dfp[dfp['is_duplicate'] == 1.0]['token_sort_ratio'][0:] , label = "1", color = 'red')

sns.distplot(dfp[dfp['is_duplicate'] == 0.0]['token_sort_ratio'][0:] , label = "0" , color = 'blue' )

plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'fuzz_ratio', data = dfp[0:] , )

plt.subplot(1,2,2)


sns.distplot(dfp[dfp['is_duplicate'] == 1.0]['fuzz_ratio'][0:] , label = "1", color = 'red')

sns.distplot(dfp[dfp['is_duplicate'] == 0.0]['fuzz_ratio'][0:] , label = "0" , color = 'blue' )

df_subsampled = dfp[0:20000]

trace = go.Scatter(
    y = df_subsampled['ctc_min'].values,
    x = df_subsampled['token_sort_ratio'].values,
    mode='markers',
    marker=dict(
        size= df_subsampled['cwc_max'].values * 25,
        color = df_subsampled['is_duplicate'].values,
        colorscale='Portland',
        showscale=True,
        opacity=0.5,
        colorbar = dict(title = 'duplicate')
    ),
    text = np.round(df_subsampled['cwc_max'].values, decimals=2)
)
data = [trace]
layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of token_sort_ratio & ctc_min',
    hovermode= 'closest',
        xaxis=dict(
        title= 'token_sort_ratio',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'ctc_min',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterWords')

print "Scatter plot shows that when token sort value and ctc_min value is low and cwc_max is low thwn most of the question pairs are non duplicated "

# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention

from sklearn.preprocessing import MinMaxScaler

dfp_subsampled = dfp[0:5000]
X = MinMaxScaler().fit_transform(dfp_subsampled[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
y = dfp_subsampled['is_duplicate'].values

from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(X)

trace1 = go.Scatter3d(
    x=tsne[:,0],
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = y,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3d embedding with engineered features')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')

