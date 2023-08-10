#import necessary modules
import pandas
import nltk
import string

#read in our data
df = pandas.read_csv("../Data/childrens_lit.csv.bz2", sep = '\t', encoding = 'utf-8', compression = 'bz2', index_col=0)
#drop missing texts
df = df.dropna(subset=['text'])
#split the text into a list
df['text_split']=df['text'].str.split()

df['word_count'] = df['text_split'].apply(len)
df['word_count']

df['title_token'] = df['title'].apply(nltk.word_tokenize)
df['title_token']

#apply the len function using .apply
df['word_count'] = df['text_split'].apply(len)
#apply the len function using lambda. This line does the same thing as line 2 above
df['word_count_lambda'] = df['text_split'].apply(lambda x: len(x))

#apply the nltk.word_tokenize function using .apply
df['title_token'] = df['title'].apply(nltk.word_tokenize)
#do the same using lambda. The next line does the same as line 7 above.
df['title_token_lambda'] = df['title'].apply(lambda x: nltk.word_tokenize(x))

df[['word_count', 'word_count_lambda','title_token', 'title_token_lambda']]

df['title_token_clean'] = df['title_token'].apply([word for word in df['title_token'] if word not in list(string.punctuation)])
df['title_token_clean']

df['title_token_clean'] = df['title_token'].apply([word for word in x if word not in list(string.punctuation)])
df['title_token_clean']

df['title_token_clean'] = df['title_token'].apply(lambda x: [word for word in x if word not in list(string.punctuation)])
df['title_token_clean']

