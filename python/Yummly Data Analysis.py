##import all the required libraries
import pandas as pd
import numpy as np
import unicodedata
import nltk
import seaborn as sns
from bs4 import BeautifulSoup
import urllib
from string import ascii_lowercase
from nltk.stem import WordNetLemmatizer
from collections import Counter
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
import ipywidgets as widgets
import operator
import pattern.en as en
from bs4 import BeautifulSoup
import urllib2
import matplotlib.pyplot as plt
from collections import Counter
from string import ascii_lowercase
import unicodedata
import nltk
import re
from nltk.stem import PorterStemmer
from gensim.models import word2vec
from sklearn import manifold
import itertools
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim import models
from gensim import similarities
import heapq

#Code to scrape BBC food ingredients
#BBC ingredients are used to standardize the ingredients in Yummly recipe dataset.

ingredients_raw = []
for alpha in ascii_lowercase:
    if alpha != 'x':
      file = urllib2.urlopen('http://www.bbc.co.uk/food/ingredients/by/letter/'+ alpha)
      soup = BeautifulSoup(file, "lxml")
      tag = soup.find('div', 'page')
      ol_tag = tag.find('ol', { "class" : "resources foods grid-view" })
      atags = ol_tag.find_all('li')
      for atag in atags:
         raw = atag.get('id')
         for letter in raw:
            if letter == "_":
                raw = raw.replace(letter," ")
         ingredients_raw.append(raw)

#Add Pluralized bbc ingredients in order to catch ingredients like tomatoes, carrots, etc
ingredients_plural = [en.pluralize(i) for i in ingredients_raw]
ingredients_raw = ingredients_raw + ingredients_plural

#Get ingredient list from the combined json file
train = pd.read_json('allrecipes.json')
train.ingredientLines.head()

#Clean recipe function to get tokenize all ingredients using nltk library & standardizing them with bbc food data
#nltk.download()

ps = PorterStemmer()

def clean_ingr(ingredients):
   
    ingr = [x.encode('UTF8') for x in ingredients] 
    tokens = nltk.word_tokenize(str(ingr))
    token_lower = [str.lower(i) for i in tokens]
    pairs = [ " ".join(pair) for pair in nltk.bigrams(token_lower)]
    clean_pairs = [i for i in pairs if i in ingredients_raw]
    pairs_joined = []
    for i in ingr:
        joined_str = re.sub('('+'|'.join('\\b'+re.escape(g)+'\\b' for g in clean_pairs)+')',lambda m: m.group(0).replace(' ', '_'),i)
        pairs_joined.append(joined_str)
    #print pairs_joined
    tokens = nltk.word_tokenize(str(pairs_joined))
    token_lower = [str.lower(i) for i in tokens]
    clean_tokens = [i for i in token_lower if i in ingredients_raw] 
    clean_ingr = clean_tokens + clean_pairs
    #print clean_ingr
    return clean_ingr

#Test clean_ingr function for first ingredient
clean_ingr(train.ingredientLines[0])

#Code to generate bag of words from the cleaned ingredient list
bags_of_words = [ Counter(clean_ingr(ingredients)) for ingredients in train.ingredientLines ]

#Find sum of every ingredient using Counter()
sumbags = sum(bags_of_words, Counter())


# Finally, plot the 10 most used ingredients
clean_df = pd.DataFrame.from_dict(sumbags, orient='index').reset_index()
clean_df = clean_df.rename(columns={'index':'ingredient', 0:'count'})
clean_df.to_csv('ingredient_clean.csv')

top_ing = clean_df.sort_values('count', ascending=False)

clean_df

ingr_only_dict = clean_df['ingredient'].to_dict()

#Plot top ingredients using bag of words
fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.barplot(x = 'ingredient', y = 'count', data = top_ing.head(10))
sns.set_palette("deep")
plt.show()
fig.savefig('ingredient_count_bag_of_words.png')

#Get all clean ingredients in list format per recipe
ingr_list = []
for ingredients in train.ingredientLines:
    ingr_list.append(clean_ingr(ingredients))

print ingr_list[0]

#PMI Calculation starts
#Point-wise mutual information to understand which ingredients go together and which ones don't

#Create combinations of ingredients
start_time = time.time()

l = []
for K in range(len(ingr_list)):
    for L in range(2,3):
        for subset in itertools.combinations(ingr_list[K], L):
            l.append(sorted(subset))       
#print("--- %s seconds ---" % (time.time() - start_time))

print l[0]

#Function - Point-wise mutual information
def pmi(dff, x, y):
    df = dff.copy()
    df['f_x'] = df.groupby(x)[x].transform('count')
    df['f_y'] = df.groupby(y)[y].transform('count')
    df['f_xy'] = df.groupby([x, y])[x].transform('count')
    df['pmi'] = np.log(len(df.index) * df['f_xy'] / (df['f_x'] * df['f_y']) )
    return df

#Convert list of tuples to dataframe
df = pd.DataFrame(l, columns = ['Ingredient1','Ingredient2'])
print df.count()

#Eliminate rows where Ingredient1 = Ingredient2
df = df[df['Ingredient1'] != df['Ingredient2']]
print df.count()

#Calculate PMI
df = pmi(df, 'Ingredient1', 'Ingredient2')
print(df.count())

df

#Eliminate rows for the same ingredient combinations
print df.pmi.count()
df = df.drop_duplicates()
print df.pmi.count()

#Sort df
df = df.sort_values('pmi',ascending='false')
df.head()

df = pd.read_csv('pmi_ingredient_similarity.csv')
df = df.sort_values('pmi',ascending=False)
top_df = df.groupby('Ingredient1').head(5)
top_df.to_csv('pmi_ingredient_similarity_top.csv')

df = df.sort_values('pmi',ascending=True)
neg_df = df.groupby('Ingredient1').head(5)
neg_df.to_csv('pmi_ingredient_similarity_neg.csv')

top_df.to_csv('pmi_ingredient_similarity.csv', encoding = 'utf-8')

temp = df[(df['I1'] == 'eel') | (df['I2'] == 'eel') ]
temp

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

##Get clean ingredients to be used as an input for word2vec model to identify ingredient similarity.

ingr_clean_df = pd.DataFrame({'Ingredient':ingr_list})
#print ingr_clean_df
print ingr_clean_df
ingr_clean_df.to_csv('ingredient_2_recipe_clean.csv')

#pip install -U gensim
#Implementing word2vec to get the recipes which are similar to each other

num_features = 300   # Word vector dimensionality                      
context = 1        # Context window size; 
downsampling = 1e-3   # threshold for configuring which higher-frequency words are randomly downsampled

# Initialize and train the model 
model = word2vec.Word2Vec(ingr_list, size=num_features, window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

flatten_list = []
for sublist in ingr_list:
    for item in sublist:
        if item not in flatten_list:
            flatten_list.append(item)

flatten_list

most_sim_list = []
corpus_sim_dict = {}
input_list = []
for i in flatten_list:
    try: 
        if len(i) > 0:
            #print i
            corpus_sim_dict.update({i:model.most_similar(i)})      
    except KeyError:
        pass

corpus_sim_dict

sim_df = pd.DataFrame([])
for key,value in corpus_sim_dict.iteritems():
    for i in value:
        sim_df = sim_df.append(pd.DataFrame({'Similar Ingredient':i[0],'Word2Vec Value': i[1],'Ingredient': key }, index=[0]), ignore_index=True)

sim_df.head()

word2vecTSNE = word2vec.Word2Vec(
    sg=1,
    size=num_features, window = context, sample = downsampling,
    min_count=3
)
word2vecTSNE.build_vocab(ingr_list)

word2vecTSNE.train(ingr_list,total_examples= word2vecTSNE.corpus_count, epochs=word2vecTSNE.iter)

if not os.path.exists("trained"):
    os.makedirs("trained")
word2vecTSNE.save(os.path.join("trained", "word2vecTSNE.w2v"))
word2vecTSNE = word2vec.Word2Vec.load(os.path.join("trained", "word2vecTSNE.w2v"))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = word2vecTSNE.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#Plot the big picture

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[word2vecTSNE.wv.vocab[word].index])
            for word in word2vecTSNE.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))
plt.show()

#Zoom in to some interesting places


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(100, 80))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

points.head(10)
points.to_csv('word2vec_tse.csv')

plot_region(x_bounds=(-40,40), y_bounds=(-40, 40))

sns.set_context("poster")
plt.show()

sim_df.to_csv('word2vec_ingredient_similarity.csv',encoding='utf-8')

df = pd.read_csv('word2vec_ingredient_similarity.csv')
df = df.sort_values('Word2Vec Value',ascending=False)
top_df = df.groupby('Ingredient').head(5)
top_df.to_csv('word2vec_ingredient_similarity_top.csv')

model.most_similar('pasta')

#testing word2vec model for a few ingredients
model.most_similar('pasta')

model.most_similar('garlic')

model.most_similar('salad')

model.most_similar('ketchup')

model.most_similar('beef')

model.most_similar('chilli')

model.most_similar('chocolate cake')

model.most_similar('lemon')

model.most_similar('garam masala')

model.most_similar('coconut oil')

model.most_similar('almond milk')

model.most_similar('bread')

model.most_similar('cumin')

model.most_similar('egg')

model.most_similar('rice')

model.most_similar('milk chocolate')

model.most_similar('mushrooms')

#Create a dictionary for all the ingredients in the recipe list

dictionary = corpora.Dictionary(ingr_list)
#print(dictionary)
#print(dictionary.token2id)
print dictionary.token2id

#Applying doc2bow on the dictionary of ingredients, which converts the ingredient to a number in every recipe
#This input format is needed for TfIdfmodel
bow_corpus = [dictionary.doc2bow(text) for text in ingr_list]
bow_corpus[0]

# train the model
tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

#print corpus for the first recipe
for i in corpus_tfidf:
    print i
    break

#print tfidf results for the first recipe
print(tfidf[bow_corpus[1]])

#Use similarities library from gensim to get the cosine similarity of the tfidf results

index = similarities.MatrixSimilarity(tfidf[bow_corpus])
index.save('ingr.index')
index = similarities.MatrixSimilarity.load('ingr.index')

sims = index[corpus_tfidf]
sims_list = [(i,j) for i,j in enumerate(sims)]

#Creating a list to hold the cosine similarity results for tfidf
tf_idf_list = []

for i,j in enumerate(sims_list):
    tf_idf_list.append(sims_list[i][1])

#Create recipe dict- to be used in creating dataframe in next step - used to decode recipe id
recipe_dict =  {k: v for k, v in enumerate(train.name)}
recipe_dict

#Use cosine similarity results to get the top 10 similar recipes for every recipe.
tf_idf_top  = []
similar_recipes_df = pd.DataFrame([])
same_item = []

#Get only top 11 largest values from the tf_idf_list - 1 recipe will be the same as itself (hence 12)
for i,item in enumerate(tf_idf_list):
    tf_idf_top.append(heapq.nlargest(11,enumerate(item), key=lambda x: x[1]))

#Remove the recipe value with 1.0 similarity - since it is the same recipe
for i,list_item in enumerate(tf_idf_top):
    for j,k in enumerate(list_item):
        if tf_idf_top[i][j][1] != 1.0:
            similar_recipes_df = similar_recipes_df.append(pd.DataFrame({'Similar_Recipe_ID': recipe_dict.get(tf_idf_top[i][j][0]),'TF-IDF Value': tf_idf_top[i][j][1],'Recipe_ID': recipe_dict.get(i)}, index=[0]), ignore_index=True)

similar_recipes_df.to_csv('similar_recipes_top_10_tf_idf.csv',encoding='utf-8')

similar_recipes_df = pd.read_csv('similar_recipes_top_10_tf_idf.csv')
similar_recipes_df = similar_recipes_df[similar_recipes_df['Recipe_ID'] != similar_recipes_df['Similar_Recipe_ID']]
print similar_recipes_df.count()

similar_recipes_df

similar_recipes_df.to_csv('similar_recipes_tf_idf.csv',encoding='utf-8')

similar_recipes_df

#Create cosine similarity matrix for all recipes 27637*27637
#Since this is a huge matrix, the top 10 similar recipe logic is a better option.

names = [i for i in range(1,len(tf_idf_list))]
final_df = pd.DataFrame.from_items(zip(names,tf_idf_list))

final_df.head()

train['recipe_id'] = train.index
recipe_name_df = train[['recipe_id','name']]
final_df['recipe_id'] = final_df.index

recipe_tf_idf_df = final_df.merge(recipe_name_df,how='left', left_on='recipe_id', right_on='recipe_id')

recipe_tf_idf_df.head(5)

#Create a list from tfidf results
#This will be used to identify ingredient importance within every recipe

corpus_list = []
for doc in corpus_tfidf:
    corpus_list.append(doc)

corpus_list[0]

#Create a flat list to eliminate repetition of ingredients and create a dict to hold the results

flat_list = []
for sublist in ingr_list:
    for item in sublist:
        if item not in flat_list:
            flat_list.append(item)
ing_dict =  {k: v for k, v in enumerate(flat_list)}

len(ing_dict)

dictionary.get(corpus_list[0][0][0])

#Create a dataframe with tf-idf values per ingredient for every recipe.
corpus_df = pd.DataFrame([])

for i,list_item in enumerate(corpus_list):
    for j,k in enumerate(list_item):
        corpus_df = corpus_df.append(pd.DataFrame({'Ingredient': dictionary.get(corpus_list[i][j][0]),'TF-IDF Value': corpus_list[i][j][1],'Recipe_ID': i}, index=[0]), ignore_index=True)

corpus_df.to_csv('ingredient_tf_idf.csv')

train['recipe_id'] = train.index
recipe_tf_idf_df = corpus_df.merge(train,how='left', left_on='Recipe_ID', right_on='recipe_id')
recipe_tf_idf_df = recipe_tf_idf_df[['Recipe_ID','name','Ingredient','TF-IDF Value']]
recipe_tf_idf_df.head(20)

recipe_tf_idf_df.to_csv('ingredient_recipe_tf_idf.csv',encoding='utf-8')

