import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_evaluation.plot import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

news = pd.read_csv('data/uci-news-aggregator.csv')
news.head()

news['CATEGORY'].unique() # unique category labels

news.CATEGORY.value_counts().plot(kind='pie', 
                                  figsize=(8,6), 
                                  fontsize=13, 
                                  autopct='%1.1f%%', 
                                  wedgeprops={'linewidth': 5}
                                  )
plt.axis('off')
plt.axis('equal')

news['TITLE'] = news['TITLE'].str.replace('[^\w\s]','').str.lower() # unpunctuate and lower case

# necessary libraries for wordcloud
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image

# create dataframe for each category
b_news = news.loc[news['CATEGORY'] == 'b'] # business
t_news = news.loc[news['CATEGORY'] == 't'] # science and technology
e_news = news.loc[news['CATEGORY'] == 'e'] # entertainment 
m_news = news.loc[news['CATEGORY'] == 'm'] # health

# convert news titles to usable strings for the word clouds
b_title = b_news['TITLE'].to_string()
t_title = t_news['TITLE'].to_string()
e_title = e_news['TITLE'].to_string()
m_title = m_news['TITLE'].to_string()

# import images and make them usable by word cloud
b_image = np.array(Image.open('images/business.jpg'))
t_image = np.array(Image.open('images/scitech.jpg'))
e_image = np.array(Image.open('images/entertainment.jpg'))
m_image = np.array(Image.open('images/health.jpg'))


fig = plt.figure(figsize=(15,15))

# setting stop-words, so words like "the" and "it" are ignored
stopwords = set(STOPWORDS)

# business news cloud
ax1 = fig.add_subplot(221)
b_wordcloud = WordCloud(background_color='white', mask=b_image, collocations=False, stopwords=stopwords).generate(b_title)
ax1.imshow(b_wordcloud, interpolation='bilinear')
ax1.set_title('business news', size=20)
ax1.axis('off')

# science and technology news cloud
ax2 = fig.add_subplot(222)
t_wordcloud = WordCloud(background_color='white', mask=t_image, collocations=False, stopwords=stopwords).generate(t_title)
ax2.imshow(t_wordcloud, interpolation='bilinear')
ax2.set_title('science & technology news', size=20)
ax2.axis('off')

# entertainment news cloud
ax3 = fig.add_subplot(223)
e_wordcloud = WordCloud(background_color='white', mask=e_image, collocations=False, stopwords=stopwords).generate(e_title)
ax3.imshow(e_wordcloud, interpolation='bilinear')
ax3.set_title('entertainment news', size=20)
ax3.axis('off')

# health news cloud
ax4 = fig.add_subplot(224)
m_wordcloud = WordCloud(background_color='white', mask=m_image, collocations=False, stopwords=stopwords).generate(m_title)
ax4.imshow(m_wordcloud, interpolation='bilinear')
ax4.set_title('health news', size=20)
ax4.axis('off')

fig.tight_layout()

# convert data to vectors
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(news['TITLE'])

y = news['CATEGORY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 30% split

# fit and score the bayesian classifier
mnb = MultinomialNB(alpha=1)
mnb.fit(X_train, y_train)
mnb.score(X_test, y_test)

confusion_matrix(y_test, mnb.predict(X_test), target_names = ['b','e','m','t']).grid(False)

sgd = SGDClassifier(n_jobs=-1, n_iter=10, random_state=1234)

# hyperparameters for tuning
sgd_grid = [{'loss': ['hinge', 'log', 'squared_hinge'],
             'alpha': [0.0001, 0.0001, 0.00001]}]

# grid search with cross validation
sgd_search = GridSearchCV(estimator=sgd, param_grid=sgd_grid, cv=5, refit=True)
sgd_search.fit(X_train, y_train)

sgd_search.best_params_

sgd_search.best_estimator_.score(X_test, y_test)

confusion_matrix(y_test, sgd_search.best_estimator_.predict(X_test), target_names = ['b','e','m','t']).grid(False)

# title-category function
def title_to_category(title):
    categories = {'b' : 'business', 
                  't' : 'science and technology', 
                  'e' : 'entertainment', 
                  'm' : 'health'}
    pridicter = sgd_search.best_estimator_.predict(vectorizer.transform([title]))
    return categories[pridicter[0]]

# sample predictions using our sgd classifier on 2017 BBC headlines
print('news title', '                                 ', 'category', '\n' 
      'Bank of England staff to go on strike', '      ', title_to_category('Bank of England staff to go on strike'), '\n'
      'Trump stance could damage Earth - Hawking', '  ', title_to_category('Trump stance could damage Earth - Hawking'), '\n'
      'Olivia de Havilland sues over TV show', '      ', title_to_category('Olivia de Havilland sues over TV show')
     )

# testing a headline from The Onion
title_to_category("Johnson & Johnson introduces 'nothing but tears shampoo' to 'toughen up' infants.")
# link to article: http://www.theonion.com/article/johnson-johnson-introduces-nothing-but-tears-shamp-2506

# another one from The Onion
title_to_category("Archaeological Dig Uncovers Ancient Race Of Skeleton People.")
# link to article: http://www.theonion.com/article/archaeological-dig-uncovers-ancient-race-of-skelet-932
# quote from article: "And though we know little of their language and means of communication, it is likely that they said 'boogedy-boogedy' a lot."

