import pickle
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_file = "preproc_data_w_text.p"
df = pickle.load(open(data_file,'rb'))

get_ipython().system('cd ../../../Shared/CMV && ls')

df.head()

df = df[df['delta_thread'] == 1]

df.columns

df.shape

comments = df['comment_content']
lengths = [len(x.split()) for x in comments]

df['comment_length'] = lengths

morality_dict = pickle.load(open('morals_dict.p', 'rb'))

#del morality_dict['morality_general']

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from collections import defaultdict

def count_moral_words(morality_dict, text):
    text = text.lower().split()
    text = [stemmer.stem(x) for x in text]
    unique_words = defaultdict(int) # counts if a word already used as there are some duplicates
    moral_count = 0
    for k,v in morality_dict.items():
        for w in v:
            unique_words[w] += 0
            wc = text.count(w)
            if unique_words[w] > 0:
                pass
            else:
                unique_words[w] = wc
                moral_count+=wc
    return moral_count

# Test
count_moral_words({'a':['sin'],'b':['sin']}, "This is an immoral sinful sentence full of hate and judgment")

#df['num_morality_tokens'] = df[commenter_morality_cols].sum(axis=1)
df['num_morality_tokens'] = [count_moral_words(morality_dict, x) for x in comments]

pickle.dump(df, open('model_df.p', 'wb'))

df['prop_moral'] = df['num_morality_tokens']/df['comment_length']

df['prop_moral'] = df['prop_moral'].fillna(0) #Fill nans with zeros

ax = df['prop_moral'].hist(bins=100)
ax.set_xlabel('Proportion of moral tokens')
ax.set_ylabel('Number of comments')

moral = df[df['num_morality_tokens'] >= 1]
not_moral = df[df['num_morality_tokens'] == 0]
has_delta = df[df['delta'] == 1]
not_delta = df[df['delta'] == 0]

plot=has_delta.prop_moral.hist(bins=50,color='black')
plot.set_xlim(0,1)
plot.set_xlabel('Proportion of moral tokens')
plot.set_ylabel('Number of comments')

plot = not_delta.prop_moral.hist(bins=50,color='red')
plot.set_xlim(0,1)
plot.set_xlabel('Proportion of moral tokens')
plot.set_ylabel('Number of comments')

plt.scatter(x=df['comment_length'],y=df['num_morality_tokens'])

p = plt.scatter(x=df['comment_length'], y=df['prop_moral'])
p.axes.set_xlabel('Number of tokens')
p.axes.set_ylabel('Proportion of moral tokens')
p.figure.set_size_inches(8,6)

df.delta.hist()



dir(fig)

#TODO: Graph % moral terms by delta.
#TODO: SHow mean delta for moral and not moral

#TODO: Current graph y axis > 1.0 because morality words not countded at the token level...



moral.shape

not_moral.shape

from scipy.stats import ttest_ind

ttest_ind(moral['delta'], not_moral['delta'])

moral['delta'].mean()

not_moral['delta'].mean()

ttest_ind(has_delta['jaccard_sim_same'], not_delta['jaccard_sim_same'])

has_delta['jaccard_sim_same'].mean()

not_delta['jaccard_sim_same'].mean()

p = has_delta['jaccard_sim_same'].hist(bins=6, color='black')
p.set_xlabel('Jaccard similarity')
p.set_ylabel('Number of OP-commenter pairs')

p = not_delta['jaccard_sim_same'].hist(bins=6, color='red')
p.set_xlabel('Jaccard similarity')
p.set_ylabel('Number of OP-commenter pairs')

p = plt.scatter(x=df['jaccard_sim_same'], y=df['comment_length'],alpha=0.003)
p.axes.set_ylabel('Comment length')
p.axes.set_xlabel('Jaccard similarity')
p.figure.set_size_inches(8,6)



