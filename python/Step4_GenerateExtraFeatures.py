import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


get_ipython().magic('matplotlib inline')

path = os.path.join('data', 'balanced_data.csv')
balanced_df = pd.read_csv(path, usecols=[1,2,3,4,5,6])

balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.head()

t = balanced_df[balanced_df.authenticity == 0].shape
f = balanced_df[balanced_df.authenticity == 1].shape

print("# of true news = {} and # of fake news = {}".format(t,f))

df = balanced_df[['title', 'author', 'text', 'authenticity']]

df['text_words_list'] = [list(x.split()) for x in df.text]
df['text_words_list']
# df.ix[0].text.split()

def image(x):
    x = x.lower()
    return x.startswith('image') or x.endswith('image')

def video(x):
    x = x.lower()
    return x.startswith('video') or x.endswith('video') 

df['Image'] = df.text_words_list.apply(lambda x: len([a for a in x if image(a)]))
df['Image'][df.Image > 0] = 1
# df['Image'][df.Image!= 1] = 0
df['Video'] = df.text_words_list.apply(lambda x: len([a for a in x if video(a)]))
df['Video'][df.Video > 0] = 1
# df['Video'][df.Video != 1] = 0
A = df['Image'][df.authenticity == 1].mean()
B = df['Image'][df.authenticity == 0].mean()
C = df['Video'][df.authenticity == 1].mean()
D = df['Video'][df.authenticity == 0].mean()
print('Fake news:Image - {}, Video - {} // Real news: Image - {}, Video - {}'.format(A,C,B,D))
print('TF-IDF deals with it')

df.author[df.author == '[]'] = 0
df.author[df.author != 0]  = 1
df.head()

df['title_list'] = [list(x) for x in df.title]
df.head()

num_cap_title = df.title_list.apply(lambda x: len([a for a in x if a.isupper()]))
len_title = df.title.str.len()
df['caprate_title'] = num_cap_title/len_title
df.caprate_title /= df.caprate_title.mean()
df.head()

f = df.caprate_title[df.authenticity == 1].mean()
t = df.caprate_title[df.authenticity == 0].mean()

print("capital rate of true news title = {} and capital rate of fake news title = {}".format(t,f))

num_exag_title = df.title_list.apply(lambda x: len([a for a in x if (a == ('!' or '?' or ':' or '-'))]))
df['exagg_puct_title'] = num_exag_title/len_title
df.exagg_puct_title /= df.exagg_puct_title.mean()
f = df.exagg_puct_title[df.authenticity == 1].mean()
t = df.exagg_puct_title[df.authenticity == 0].mean()

print("exaggerating punctuation rate of true news title = {} and exaggerating punctuation rate of fake news title = {}".format(t,f))

df['text_list'] = [list(x) for x in df.text]
num_cap_text = df.text_list.apply(lambda x: len([a for a in x if a.isupper()]))
len_text = df.text.str.len()
df['caprate_text'] = num_cap_text/len_text
df.caprate_text /= df.caprate_text.mean()
f = df.caprate_text[df.authenticity == 1].mean()
t = df.caprate_text[df.authenticity == 0].mean()

print("capital rate of true news text = {} and capital rate of fake news text = {}".format(t,f))

selected_df = df[['author', 'caprate_title', 'exagg_puct_title']]
selected_df.head()

print(df.exagg_puct_title.mean(),df.caprate_title.mean() )

df.head()

A = df.author.std()
B = df.caprate_title.std()
C = df.exagg_puct_title.std()

print(A,B,C)

D = df.author.max()
E = df.caprate_title.max()
F = df.exagg_puct_title.max()
print(D,E,F)

bar_df = pd.DataFrame(columns = ['Real_news_mean', 'Real_news_sterr', 'Fake_news_mean', 'Fake_news_sterr'], index = ['author','caprate_title', 'exagg_puct_title'])
bar_df.Real_news_mean = [df.author[df.authenticity == 0].mean(), df.caprate_title[df.authenticity == 0].mean(), df.exagg_puct_title[df.authenticity == 0].mean() ]
bar_df.Fake_news_mean = [df.author[df.authenticity == 1].mean(), df.caprate_title[df.authenticity == 1].mean(), df.exagg_puct_title[df.authenticity == 1].mean() ]
bar_df.Real_news_sterr =[df.author[df.authenticity == 0].std()/np.sqrt(df.shape[0]), df.caprate_title[df.authenticity == 0].std()/np.sqrt(df.shape[0]), df.exagg_puct_title[df.authenticity == 0].std()/np.sqrt(df.shape[0]) ]
bar_df.Fake_news_sterr =[df.author[df.authenticity == 1].std()/np.sqrt(df.shape[0]), df.caprate_title[df.authenticity == 1].std()/np.sqrt(df.shape[0]), df.exagg_puct_title[df.authenticity == 1].std()/np.sqrt(df.shape[0]) ]
bar_df

f, ax = plt.subplots(1,1)
ax.legend(labels = ['Real news', 'Fake news'], loc = 'upper right', bbox_to_anchor=(1.2, 0.5))
width = 0.25
groups = [1,2,3]
ax.set_ylabel('Normarlized Correlation Ratio',fontsize = 15)
ax.set_title('Example of a few features other than tf-idf, and their correlation with authenticity', fontsize = 15)
ax.set_xticks(groups)
ax.set_xticklabels(('Authors', 'Rate of uppercases\nin title', 'Rate of exaggerating punction\nin title'), fontsize = 15,rotation = 45)


c = (x for x in ['red','green'])
for a in ['Real_news', 'Fake_news']:
    ax.bar(groups, bar_df[a+'_mean'], width, color=next(c), alpha = 0.4, yerr=bar_df[a+'_sterr'])
    groups = [(x + width) for x in groups]

plt.savefig('additional_bar.jpeg', bbox_inches = 'tight')

path = os.path.join('data','additional_features.csv')
selected_df.to_csv(path)

