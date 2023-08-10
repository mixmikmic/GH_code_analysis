#import libraries and set seaborn styling
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tmdbsimple as tmdb
import requests
import pandas as pd
import time
import numpy as np
from ast import literal_eval
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
sns.set_context('talk')
sns.set_style('ticks')

import json
id_to_genre = json.load(open('data/id_to_genre.json'))

id_to_genre = {int(key):value for key, value in id_to_genre.items()} #convert string keys to int keys
id_to_genre

movies = pd.read_csv('data/movies.csv', encoding='utf-8', converters={'tmdb_genres':literal_eval})

movies.release_date = pd.to_datetime(movies.release_date)
movies.head()

print('Title:\n', movies.title[12],
      '\n\nTMDB Plot:\n', movies.tmdb_plot[12], 
      '\n\nIMDB Plot:\n', movies.imdb_plot[12])

plt.hist(movies.vote_average)
plt.title('Movie Ratings (0-10 scale)')
plt.xlabel('Average Rating')
plt.ylabel('Count')
sns.despine()

genre_list_ints = []
for sublist in list(movies.tmdb_genres):
    for item in sublist:
        genre_list_ints.append(item)

genre_list_strings = []
for genre in genre_list_ints:
    genre_list_strings.append(id_to_genre[genre])
    
genre_counts = Counter(genre_list_strings)
genre_df = pd.DataFrame.from_dict(genre_counts, orient='index')
genre_df.sort_values(by = 0, ascending=False).plot(kind='bar', legend=False)
plt.title('TMDB Genre Counts in Dataset')
plt.ylabel('Count')
plt.xlabel('Genre')
sns.despine()

#count number of genres per movie
tmdb_num_genres_per_movie = [len(genres) for genres in movies.tmdb_genres]

#collect counts and plot for both imdb and tmdb
tmdb_num_genre_counts = Counter(tmdb_num_genres_per_movie)
tmdb_genre_counts_df = pd.DataFrame.from_dict(tmdb_num_genre_counts, orient='index')
tmdb_genre_counts_df = tmdb_genre_counts_df.sort_index()
tmdb_genre_counts_df.plot(kind='bar', legend=False)
plt.title('Genres per TMDB Movie')
plt.xlabel('Genres Labels per Movie')
plt.ylabel('Count')
sns.despine()

print('Average TMDB Genres:', np.mean(tmdb_num_genres_per_movie))

movie_years = [year.year for year in movies.release_date]
year_counts = Counter(movie_years)
year_df = pd.DataFrame.from_dict(year_counts, orient='index')
year_df.sort_index().plot(kind='bar', legend=False)
plt.title('Movies Counts by Year in Dataset')
plt.tick_params(labelsize=9)
plt.ylabel('Total Movies')
plt.xlabel('Year')
sns.despine()

#tmdb lengths
tmdb_plot_lengths = []
for plot in movies.tmdb_plot:
    plot_length = len(plot.split())
    tmdb_plot_lengths.append(plot_length)
    
#imdb lengths
imdb_plot_lengths = []
for plot in movies.imdb_plot:
    plot_length = len(plot.split())
    imdb_plot_lengths.append(plot_length)
    
#plot tmdb lengths
bins = np.linspace(0, 500, 30)
plt.hist(tmdb_plot_lengths, bins, alpha = 0.8, label = 'TMDB')
plt.hist(imdb_plot_lengths, bins, alpha = 0.8, label = 'IMDB')
plt.xlabel('Plot Length (words)')
plt.ylabel('Count')
plt.title('Distribution of TMDB & IMDB Movie Plot Lengths')
plt.legend(loc='lower right')
sns.despine();

#scatter of TMDB vs IMDB plot lengths
plt.scatter(imdb_plot_lengths, tmdb_plot_lengths)
plt.title('TDMB vs. IMDB Plot Lenghts')
plt.xlabel('IMDB Plot Lengths')
plt.ylabel('TMDB Plot Lengths')
sns.despine();

print('Shortest and longest plots (words):')
print('TMDB: ', min(tmdb_plot_lengths),',', max(tmdb_plot_lengths))
print('Mean TMDB plot length:', np.mean(tmdb_plot_lengths))
print()
print('IMDB: ', min(imdb_plot_lengths),',', max(imdb_plot_lengths))
print('Mean IMDB plot length:', np.mean(imdb_plot_lengths))

sns.boxplot(tmdb_num_genres_per_movie, tmdb_plot_lengths)
plt.title('TMDB Plot Lengths vs. Number of Genres per Movie')
plt.xlabel('Genres per Movie')
plt.ylabel('Plot Length (words)')
sns.despine()

movies['combined_plots'] = movies['tmdb_plot'] + ' ' + movies['imdb_plot']

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.005, 
                          stop_words = get_stop_words('en'))

tmdb_bow=tfidf_vectorizer.fit_transform(movies['tmdb_plot'])
imdb_bow=tfidf_vectorizer.fit_transform(movies['imdb_plot'])
combined_bow = tfidf_vectorizer.fit_transform(movies['combined_plots'])

#check shape of resulting matrices
tmdb_bow.shape, imdb_bow.shape, combined_bow.shape

#each plot is a row vector of the TFIDF sparse matrices
movies['tmdb_bow_plot'] = [plot for plot in tmdb_bow]
movies['imdb_bow_plot'] = [plot for plot in imdb_bow]
movies['combined_bow_plots'] = [plot for plot in combined_bow]

#resave movies csv
movies.to_csv('data/movies.csv', encoding="utf-8", index=False)

#save TFIDF matrices an numpy arrays
np.save('data/tmdb_bow.npy',tmdb_bow.toarray())
np.save('data/imdb_bow.npy',imdb_bow.toarray())
np.save('data/combined_bow.npy', combined_bow.toarray())

#define tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#set stop words list
english_stop = get_stop_words('en')
print(len(english_stop))

#function to clean plots
def clean_plot(plot):
    '''
    clean_plot()
    -applies the following the plot of a movie:
        1) lowers all strings
        2) tokenizes each word
        3) removed English stop words

    -inputs: plot (string)
    
    -outputs: list representation of plot
    '''
    plot = plot.lower()
    plot = tokenizer.tokenize(plot)
    plot = [word for word in plot if word not in english_stop]
    return plot

#check function on first movie's clean plot
print(movies.tmdb_plot[0])
print(clean_plot(movies.tmdb_plot[0]))

#apply to movies df for both imdb and tmdb
movies['tmdb_clean_plot'] = movies['tmdb_plot'].apply(lambda x: clean_plot(x))
movies['imdb_clean_plot'] = movies['imdb_plot'].apply(lambda x: clean_plot(x))
movies['combined_clean_plot'] = movies['combined_plots'].apply(lambda x: clean_plot(x))

#check some outputs
movies.tmdb_clean_plot[1:5], movies.imdb_clean_plot[1:5], movies.combined_clean_plot[1:5]

#Load the pretrained google news word2vec model
model = models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

#collect words not in the Google News w2v model
not_w2v = []

#word2vec function
def apply_words2Vec(cleaned_plot, mean=False):
    
    """
    apply_words2Vec()
    -applies the following transformations to the cleaned plot of a movie:
        1) removes words that are not in google's model
        2) creates a 300-dimension vector representation of each word
        3) outputes vector of vectors for plot
        If mean = True
        4) converts the resulting nd_array into a 1d_array via np.mean() and
           outputs single vector for each plot.
    -also keeps track of all words not found in google's model
    
    -inputs: cleaned_plot (string)
    
    -outputs: vector representation of plot
    
    """
    vecs=[]
    for word in cleaned_plot:
        #add word vector to list if it is in the google model
        try:
            vecs.append(model.word_vec(word)) 
        except:
            #if the word is not in the w2v model, add it to
            #our list of skipped words
            not_w2v.append(word)
    
    #take the column-wise mean of vecs to reduce nd_aray to 1d_array
    if mean == True:
        vecs = np.mean(vecs, axis=0)
        return vecs
    #return matrix of w2v arrays where each row is a word in the plot
    return np.stack(vecs)

#apply transformation to three sets of plots and add columns to df

#columns with mean w2v
movies['tmdb_w2v_plot_mean'] = movies['tmdb_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=True))
movies['imdb_w2v_plot_mean'] = movies['imdb_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=True))
movies['combined_w2v_plot_mean'] = movies['combined_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=True))

#columns with w2v matrix for each plot
movies['tmdb_w2v_plot_matrix'] = movies['tmdb_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=False))
movies['imdb_w2v_plot_matrix'] = movies['imdb_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=False))
movies['combined_w2v_plot_matrix'] = movies['combined_clean_plot'].apply(lambda x: apply_words2Vec(x, mean=False))

#check shapes of first movie vectors to confirm nd_array and 300-dimensions
print('Mean vector representations:')
print(movies.loc[0,'tmdb_w2v_plot_mean'].shape, 
      movies.loc[0,'imdb_w2v_plot_mean'].shape, 
      movies.loc[0, 'combined_w2v_plot_mean'].shape)

print('Matrix representations:')
print(movies.loc[0,'tmdb_w2v_plot_matrix'].shape, 
      movies.loc[0,'imdb_w2v_plot_matrix'].shape,
      movies.loc[0,'combined_w2v_plot_matrix'].shape)

#print some of our skipped words
print(len(not_w2v))
np.random.seed(112)
print(np.random.choice(not_w2v, 50, replace=False))

for plot in movies.tmdb_w2v_plot_mean:
    if len(plot) != 300:
        print("AH! no word2vec representation")
print('All TMDB movies have a word2vec representation.')

for plot in movies.imdb_w2v_plot_mean:
    if len(plot) != 300:
        print("AH! no word2vec representation")
print('All IMDB movies have a word2vec representation.')

#w2v mean vectors
np.save('data/tmdb_w2v_mean.npy', movies['tmdb_w2v_plot_mean'].as_matrix())
np.save('data/imdb_w2v_mean.npy', movies['imdb_w2v_plot_mean'].as_matrix())
np.save('data/combined_w2v_mean.npy', movies['combined_w2v_plot_mean'].as_matrix())

#w2v matrices
np.save('data/tmdb_w2v_matrix.npy', movies['tmdb_w2v_plot_matrix'].as_matrix())
np.save('data/imdb_w2v_matrix.npy', movies['imdb_w2v_plot_matrix'].as_matrix())
np.save('data/combined_w2v_matrix.npy', movies['combined_w2v_plot_matrix'].as_matrix())

movies.to_csv('data/movies.csv', encoding="utf-8", index=False)

#turn plots into TaggedDocuments
#apply to movies df for imdb, tmdb, and combined
cols = ['tmdb_clean_plot', 'imdb_clean_plot', 'combined_clean_plot']
n_cols = ['tmdb_tag_docs', 'imdb_tag_docs', 'combined_tag_docs']
for ncol, col in zip(n_cols, cols):
    acc = []
    for i, plot in enumerate(movies[col]): 
        z = models.doc2vec.TaggedDocument(words=plot, tags=[str(i)])
        acc.append(z)
    movies[ncol] = acc
movies['tmdb_tag_docs'][:1]

cols = ['tmdb_tag_docs', 'imdb_tag_docs', 'combined_tag_docs']
n_cols = ['tmdb_doc_vec', 'imdb_doc_vec', 'combined_doc_vec']
for ncol, col in zip(n_cols, cols):
    model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.001, vector_size=19, dm=0, window=5)
    model.build_vocab(movies[col])
    model.train(movies[col], total_examples=999, epochs=20)
    movies[ncol] = [model.docvecs[x] for x in range(999)]
    
movies['combined_doc_vec'][0]

np.save('data/tmdb_doc_vec.npy', movies['tmdb_doc_vec'].as_matrix())
np.save('data/imdb_doc_vec.npy', movies['imdb_doc_vec'].as_matrix())
np.save('data/combined_doc_vec.npy', movies['combined_doc_vec'].as_matrix())

