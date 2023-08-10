import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

imdb_raw = pd.read_csv('imdb_labelled.txt', '\t', header=None)
imdb_raw.columns = ['text', 'sentiment']

#Bad review keywords

#I randomnly chose these words from the bad reviews from the imdb_labelled.txt file

keywords = ['aimless', 'disappointed', 'lacks', 'wasted', 'mediocre', 'unfunny', 'predictable', 
            'obvious', 'lame', 'far', 'worst', 'slow', 'boring', 'bored', 'bad', 'awful', 'terrible', 
            'waste', 'negative', 'walked out', 'flat', 'problem', 'below average', 'long', 'little', 
            'aimless', 'insulting', 'avoid', 'hard to watch', 'annoying', 'annoyed', 'annoyingly','boredom', 
            'crap', 'bs', 'crazy', 'disapprove', 'disgusted', 'disgust', 'embarrassing', 'embarrass', 
            'god-awful', 'hate', 'hated', 'horrid', 'idiots', 'idiot', 'ill-advised', 'ill-conceived', 
            'implausible', 'impossible', 'inaccurate', 'infuriate', 'junk', 'lacked', 'mad', 'miss', 'offend', 
            'offended', 'overdone', 'overplayed', 'retard', 'retarded', 'rotten', 'shit', 'silly', 
            'slow-moving', 'stink', 'stunk', 'sub-par', 'tank', 'tanked', 'trash', 'unhappy', 'upset', 
            'one-dimensional', "can't recommend" ]

#Columns to identify if keyword in review
for key in keywords:
    imdb_raw[str(key)] = imdb_raw.text.str.contains(
        str(key), 
        case=False
    )

#Heatmap to show independence
plt.figure(figsize=(15, 15))
sns.heatmap(imdb_raw.corr())

#x and y values
data = imdb_raw[keywords]
target = imdb_raw['sentiment']

# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))

#Repeating with Amazon data

amzn = pd.read_csv('amazon_cells_labelled.txt', '\t', header=None)
amzn.columns = ['text', 'sentiment']

#Columns to identify if keyword in review
for key in keywords:
    amzn[str(key)] = amzn.text.str.contains(
        str(key), 
        case=False
    )

#x and y values
data = amzn[keywords]
target = amzn['sentiment']

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))

#Repeating with Yelp data

yelp = pd.read_csv('yelp_labelled.txt', '\t', header=None)
yelp.columns = ['text', 'sentiment']

#Columns to identify if keyword in review
for key in keywords:
    yelp[str(key)] = yelp.text.str.contains(
        str(key), 
        case=False
    )

#x and y values
data = yelp[keywords]
target = yelp['sentiment']

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))

