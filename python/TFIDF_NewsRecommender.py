PATH_NEWS_ARTICLES="/home/phoenix/Documents/HandsOn/Final/news_articles.csv"
ARTICLES_READ=[2,7]
NUM_RECOMMENDED_ARTICLES=5

try:
    import numpy
    import pandas as pd
    import pickle as pk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    from nltk.stem.snowball import SnowballStemmer
    import nltk
    stemmer = SnowballStemmer("english")
except ImportError:
    print('You are missing some packages! '           'We will try installing them before continuing!')
    get_ipython().system('pip install "numpy" "pandas" "sklearn" "nltk"')
    import numpy
    import pandas as pd
    import pickle as pk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    from nltk.stem.snowball import SnowballStemmer
    import nltk
    stemmer = SnowballStemmer("english")
    print('Done!')

news_articles = pd.read_csv(PATH_NEWS_ARTICLES)
news_articles.head()

#Select relevant columns and remove rows with missing values
news_articles = news_articles[['Article_Id','Title','Content']].dropna()
#articles is a list of all articles
articles = news_articles['Content'].tolist()
articles[0] #an uncleaned article

def clean_tokenize(document):
    document = re.sub('[^\w_\s-]', ' ',document)       #remove punctuation marks and other symbols
    tokens = nltk.word_tokenize(document)              #Tokenize sentences
    cleaned_article = ' '.join([stemmer.stem(item) for item in tokens])    #Stemming each token
    return cleaned_article

cleaned_articles = map(clean_tokenize, articles)
cleaned_articles[0]  #a cleaned, tokenized and stemmed article 

#Get user representation in terms of words associated with read articles
user_articles = ' '.join(cleaned_articles[i] for i in ARTICLES_READ)

user_articles

#Generate tfidf matrix model for entire corpus
tfidf_matrix = TfidfVectorizer(stop_words='english', min_df=2)
article_tfidf_matrix = tfidf_matrix.fit_transform(cleaned_articles)
article_tfidf_matrix #tfidf vector of an article

#Generate tfidf matrix model for read articles
user_article_tfidf_vector = tfidf_matrix.transform([user_articles])
user_article_tfidf_vector

user_article_tfidf_vector.toarray()

articles_similarity_score=cosine_similarity(article_tfidf_matrix, user_article_tfidf_vector)

recommended_articles_id = articles_similarity_score.flatten().argsort()[::-1]

recommended_articles_id

#Remove read articles from recommendations
final_recommended_articles_id = [article_id for article_id in recommended_articles_id 
                                 if article_id not in ARTICLES_READ ][:NUM_RECOMMENDED_ARTICLES]

final_recommended_articles_id

#Recommended Articles and their title
print 'Articles Read'
print news_articles.loc[news_articles['Article_Id'].isin(ARTICLES_READ)]['Title']
print '\n'
print 'Recommender '
print news_articles.loc[news_articles['Article_Id'].isin(final_recommended_articles_id)]['Title']



