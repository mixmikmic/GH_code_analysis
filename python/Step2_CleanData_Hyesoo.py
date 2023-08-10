import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)

fake_df, true_df = pd.read_csv('hyesoo_fake_news_rawdata.csv'), pd.read_csv('hyesoo_true_news_rawdata.csv')

# fake_news
fake_df.info()

fake_df.head(n=30)

fake_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
fake_df = fake_df[fake_df.text.notnull()]
fake_df = fake_df[(fake_df.text.str.len()) > 500]
fake_df.reset_index(inplace = True, drop = True)

list(set(list(fake_df.source)))
# print(len(list(set(list(fake_df.source)))))

## Useful functions

# length of the text 
def text_length (df, source):
    text_length = {}
    for index in list(df[df.source == source].index):
        text_length[index] = len(df.ix[index]['text'])
    return text_length

check_list_fake = []
check_list_true = []
fake_delete_list = []
true_delete_list = []
print(fake_df.shape[0])

# bizstandardnews

fake_df[fake_df.source == 'bizstandardnews']

text_length (fake_df, 'bizstandardnews')

fake_df.ix[91].text

check_list_fake += ['bizstandardnews - fine']

#Conservativedailypost

fake_df[(fake_df.source == 'Conservativedailypost')]

text_length (fake_df, 'Conservativedailypost')

check_list_fake += ['Conservativedailypost - fine']

#americanfreepress

fake_df[(fake_df.source == 'americanfreepress')]

text_length (fake_df, 'americanfreepress')

fake_df.ix[696].text

fake_delete_list += list(fake_df[(fake_df['source'] =='americanfreepress') & (fake_df.text.str.len()<1000)].index)

check_list_fake += ['americanfreepress - deleted a bunch']

#aurora-news
fake_df[fake_df.source == 'aurora-news']

text_length (fake_df, 'aurora-news')

fake_delete_list += [x for x in range(785,811) if x != 799]

check_list_fake += ['aurora-news - deleted all except for one']

#Clashdaily
fake_df[fake_df.source == 'Clashdaily']

text_length (fake_df, 'Clashdaily')

fake_df.ix[860].text

fake_delete_list += [833,847]

check_list_fake += ['Clashdaily - have some short articles, but just delete a few because they are meaningful']

#Bighairynews
fake_df[fake_df.source == 'Bighairynews'] 

text_length (fake_df, 'Bighairynews')

fake_df.ix[83].text

check_list_fake += ['Bighairynews - have some short articles but not deleting any']

#Americannews

fake_df[fake_df.source == 'Americannews']

text_length (fake_df, 'Americannews')

fake_df.ix[22].text

check_list_fake += ['Americannews - not deleting any']

#ABCnews

fake_df[fake_df.source == 'ABCnews']

text_length (fake_df, 'ABCnews')

check_list_fake += ['ABCnews - not deleting any']

#ddsnewstrend

fake_df[fake_df.source == 'ddsnewstrend']

text_length (fake_df, 'ddsnewstrend')

fake_df.ix[556].text

check_list_fake += ['ddsnewstrend - not deleting any']

#bipartisanreport
fake_df[fake_df.source == 'bipartisanreport']

text_length (fake_df, 'bipartisanreport')

check_list_fake += ['bipartisanreport - not deleting any']

#Americanoverlook
fake_df[fake_df.source == 'Americanoverlook']

text_length (fake_df, 'Americanoverlook')

check_list_fake += ['Americanoverlook - not deleting any']

#beforeitsnews
fake_df[fake_df.source == 'beforeitsnews']

text_length (fake_df, 'beforeitsnews')

fake_df.ix[318].text

fake_delete_list += [306]

check_list_fake += ['beforeitsnews - have some short articles and deleting one']

#wordpress
fake_df[fake_df.source == 'wordpress']

text_length (fake_df, 'wordpress')

fake_df.ix[288].text

fake_delete_list += [126,150,155]

check_list_fake += ['wordpress - deleted a few']

check_list_fake

fake_df.drop(fake_df.index[fake_delete_list], inplace = True)
fake_df.drop_duplicates('text', inplace = True)
fake_df.reset_index(inplace = True, drop = True)

len(check_list_fake)

fake_df.shape[0]

fake_df.head(n = 50)

# true_news

true_df.info()

true_df.head(n=30)

true_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
true_df = true_df[true_df.text.notnull()]
true_df = true_df[(true_df.text.str.len()) > 500]
true_df.drop_duplicates('text', inplace = True)
true_df.reset_index(inplace = True, drop = True)

true_df.info()

list(set(list(true_df.source)))

#nytimes
true_df[true_df.source == 'nytimes']

text_length (true_df, 'nytimes')

true_df[(true_df.source == 'nytimes') & (true_df.text.str.len()<1500)]

true_df.ix[138].text

true_delete_list += [34,36,37,39,41]

check_list_true += ['nytimes - deleted a few']

#foxnews
true_df[true_df.source == 'foxnews']

text_length (true_df, 'foxnews')

true_df[(true_df.source == 'foxnews') & (true_df.text.str.len()<800)]

true_df.ix[802].url

true_delete_list += list(true_df[(true_df.source == 'foxnews') & (true_df.text.str.contains('Audio clip'))].index)

true_df.ix[830].text

check_list_true += ['foxnews - deleted audio clips']

#npr
true_df[true_df.source == 'npr']

text_length (true_df, 'npr')

true_df[(true_df.source == 'npr') & (true_df.text.str.len()<1200)]

true_df.ix[392].text

true_delete_list += list(true_df[(true_df.source=='npr')&(true_df.url.str.contains('ethics'))].index)
true_delete_list += [397]

check_list_true += ['npr - delete ethics blabla']

#bbc
true_df[true_df.source == 'bbc']

text_length (true_df, 'bbc')

true_df[(true_df.source == 'bbc') & (true_df.text.str.len()<1200)]

true_df.ix[159].text

check_list_true += ['bbc - delete nothing']

#reuters
true_df[true_df.source == 'reuters']

true_delete_list += list(range(506,521)) + list(range(557,564)) # non-english

text_length (true_df, 'reuters')

true_df[(true_df.source == 'reuters') & (true_df.text.str.len()<1000)]

true_df.ix[501].text

check_list_true += ['reuters']

#cnn
true_df[true_df.source == 'cnn']

text_length (true_df, 'cnn')

true_df[(true_df.source == 'cnn') & (true_df.text.str.len()<1000)]

##### maybe???? Get rid of the articles 'Read More'
# true_delete_list += list(true_df[true_df.text.str.contains('Read More')].index)

check_list_true += ['cnn - may want to check more later']

#politico
true_df[true_df.source == 'politico']

text_length (true_df, 'politico')

true_df[(true_df.source == 'politico') & (true_df.text.str.len()<1000)]

true_df.ix[1077].text

check_list_true += ['politico - no need to delete any']

true_df.drop(true_df.index[true_delete_list], inplace = True)
true_df.reset_index(inplace = True, drop = True)
true_df.shape[0]

true_df['authenticity'] = 1
fake_df['authenticity'] = 0

hyesoo_df = pd.concat([true_df, fake_df], join = 'outer')

hyesoo_df.head()

hyesoo_df.tail()

hyesoo_df = hyesoo_df.sample(frac=1).reset_index(drop=True)

hyesoo_df.info()

hyesoo_df.tail(n = 30)

hyesoo_df.to_csv('hyesoo_df.csv')



