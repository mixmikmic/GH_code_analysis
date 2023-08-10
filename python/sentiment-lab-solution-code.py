import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

tweets = pd.read_csv("../dataset/Tweets.csv")

tweets.head()

tweets['airline_sentiment'].value_counts() / len(tweets)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

compound= []
neg = []
neu = []
pos = []
for tweet in tweets['text']:
    sent = sia.polarity_scores(tweet)
    compound.append(sent['compound'])
    neg.append(sent['neg'])
    neu.append(sent['neu'])
    pos.append(sent['pos'])

tweets['compound'] = compound
tweets['neg'] = neg
tweets['neu'] = neu
tweets['pos'] = pos

tweets.head()

y = tweets['airline_sentiment']
X = tweets[['compound', 'neg', 'neu', 'pos']]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
from sklearn.model_selection import cross_val_score, train_test_split

rf.fit(X,y)

cross_val_score(rf, X, y)
# versus the baseline (63%), this is a little weak.

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.30,
                                                        random_state=14)
rf.fit(X_train,y_train)


conmat = np.array(confusion_matrix(y_test, rf.predict(X_test)))
confusion = pd.DataFrame(conmat, index=['negative', 'neutral', 'positive'],                     columns=['Pred neg', 'Pred neutral', 'Pred pos'])

plt.figure(figsize = (6,6))
heat = sns.heatmap(confusion, annot=True, annot_kws={"size": 20},cmap='Blues',fmt='g', cbar=False)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
plt.title("Confusion Matrix", fontsize=20)

print classification_report(y_test, rf.predict(X_test))

