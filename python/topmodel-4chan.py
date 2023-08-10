import pandas as pd

# read data
df = pd.read_csv("posts_4chan_pol.csv")

# creat new column where 'Comments' strings are split into list-
# this is needed as input for Gensim dictionaries
df['Comment_split'] = df.Comment.str.split()

# clean up Comment_split column:

# drop N/A values
df = df.dropna()

# define function to remove integers from lists
def rm_integer( list ):
    
    print(list)
    newlist = []
        
    for s in list:
        if (s.isdigit() or (s[0] == '-' and s[1:].isdigit())):
            continue
        else:
            newlist.append(s)
    
    return newlist
    
# apply remove-integer function to dataframe column
df['Comment_split'] = df['Comment_split'].apply(rm_integer)

#define function to remove common words from lists
from nltk.corpus import stopwords
def rm_commonwords(list) :
    addl_words_to_remove = ["u","it"]
    newlist = [ ]
    for s in list:
        if s in stopwords.words('english'):
            continue
        elif s in addl_words_to_remove:
            continue
        else:
            newlist.append(s)
            
    return newlist
    
#apply remove common words function to dataframe column
df['Comment_split'] = df['Comment_split'].apply(rm_commonwords)
print (df['Comment_split'].head(50))

# Importing Gensim (see tutorial)
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(df['Comment_split'])

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in df['Comment_split']]


# Do Topic Modeling (see tutorial)
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

# print output
topics = (ldamodel.print_topics(num_topics=10, num_words=4))

for key, val in topics:
    print(val.encode('utf-8'))



