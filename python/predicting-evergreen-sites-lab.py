import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
get_ipython().magic('matplotlib inline')

# set max printout options for pandas:
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 300

evergreen_tsv = './datasets/evergreen_sites.tsv'

# A:
df = pd.read_csv(evergreen_tsv, sep='\t')
df.shape

df.info()

# clean is_news column
df['is_news'].unique()

# replace ? with 0
df.loc[:,'is_news'] = df['is_news'].map(lambda x: 0 if x == '?' else int(x))
df['is_news'].value_counts()

# check alchemy category score
df['alchemy_category_score'].value_counts().sort_values(ascending=False)

# check those rows with '?'
# now we know that the score is 0 because there is no category
print(df[df['alchemy_category_score'] == '?'].alchemy_category.value_counts())

# Since our concern is to predict greeness of content, lets just drop the alchemy category score column
# Column might be useful for us if we intent to predict category scores
new_df = df.drop(labels='alchemy_category_score', axis=1)

# replace '? in alchemy_categoy to 'unknown'
new_df.loc[:,'alchemy_category'] = new_df['alchemy_category'].map(lambda x: 'unknown' if x == '?' else x)

# check news_front_page column
new_df['news_front_page'].value_counts()

# lets just drop those rows with '?'
index_to_drop = new_df[new_df['news_front_page']=='?'].index.tolist()
neww_df = new_df.drop(labels=index_to_drop)
neww_df.loc[:,'news_front_page'] = neww_df['news_front_page'].astype('int64')

# Now lets get two new columns from boiler plate, title & body
# first we convert boilerplate values into python dicts
bp_dict = neww_df.boilerplate.map(lambda x: json.loads(x))
'title' in bp_dict.values[0].keys()

# now we make 2 new copies of it, 1 for title and the other for body
title_dict = bp_dict.copy(deep=True)
body_dict = bp_dict.copy(deep=True)

# extract title into title column else put Nan
# do the same for body
# then combine them with main dataframe
neww_df['title'] = title_dict.map(lambda x: x['title'] if 'title' in x.keys() else np.nan)
neww_df['body'] = body_dict.map(lambda x: x['body'] if 'body' in x.keys() else np.nan)

neww_df.isnull().sum().sort_values(ascending=False)

# remove the nan rows for title and body
newww_df = neww_df.dropna()

# check whether binary columns are really binary
binary_cols = ['framebased','hasDomainLink','is_news','lengthyLinkDomain','news_front_page','label']
for col in binary_cols:
    print(newww_df[col].value_counts())
    print('')

# final check using describe
newww_df.describe()

# drop framebased since all 0, unable to predict anything with it
newwww_df = newww_df.drop(labels='framebased',axis=1)

# A:
newwww_df[newwww_df['label'] == 1].head(2)

newwww_df[newwww_df['label'] == 0].head(2)

# A:
sns.countplot(x='is_news', hue='label', data=newwww_df[['is_news','label']])

# Apparently not, if they are we should observe a huge difference between evergreen or not when sites is a news

import statsmodels.formula.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# A:
news_data = newwww_df[['label','is_news']]
formula = 'label ~ is_news'
news_model = sm.logit(formula, data=news_data).fit()
news_model.summary()

# A:
# effect of site being news is statistically insignificant on evergreeness of site
# Accept Null hypothesis that news site and non-news site have equal probability of being evergreen

# A:
# check probability of site being evergreen for each website category
newwww_df.groupby(['alchemy_category'])['label'].mean()

# for each category, score > 0.5 means probability of category being evergreen is higher

# now we plot countplot
category_data = newwww_df[['alchemy_category','label']]
fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(x='alchemy_category', hue='label', data=category_data, ax=ax)

# Seems like website category does affect green-ness of site

# A:
formula = "label ~ C(alchemy_category, Treatment(reference='unknown'))"
log_model = sm.logit(formula, data=category_data).fit(method='bfgs', maxiter=1000)

log_model.summary()

# A:
# Categories must be read as significantly different from unknown or not
# categories that are insignificant are the following:
# - weather
# - science technology
# - religion
# - law crime
# - culture politics

# Among the categories that are significant:

# Positive predictors of greeness
# - Business
# - Health
# - Recreation

# Negative predictors of greeness
# - Arts Entertainment
# - Computer Internet
# - Gaming
# - Sports

# A:
image_data = newwww_df[['label','image_ratio']]
image_data.head()

# cut the image_ratio into 5 bins in ascending value
pd.qcut(image_data['image_ratio'], 5).value_counts().sort_values(ascending=False)

image_data['image_ratio_binned'] = pd.qcut(image_data['image_ratio'], 5)
image_data.head(5)

image_data.info()

# Now we can take a look at the greeness for each category of image_ratio
fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(x='image_ratio_binned', hue='label', data=image_data, ax=ax)

fig, ax = plt.subplots(figsize=(20,5))
sns.factorplot(x='image_ratio_binned', y='label', data=image_data, ax=ax)

# quadratic curve means we have to come up with a quadratic equation for our regression

# A: 
formula = 'label ~ image_ratio + np.power(image_ratio,2)'
model = sm.logit(formula, data=image_data).fit()
model.summary()

# something wrong here, linear function should have positive effect while quadratic should have negative effect

image_data['image_ratio_pctl'] = image_data.image_ratio.map(lambda x: stats.percentileofscore(image_data.image_ratio.values, x))

formula = 'label ~ image_ratio_pctl + np.power(image_ratio_pctl,2)'
model = sm.logit(formula, data=image_data).fit()
model.summary()

# now it looks like our EDA hahah

# A:

# Positive effect from image ratio

# Negative quadratic effect of image ratio. Which means at a certain point the curve starts to turn downwards.
# From this we can tell that the median values has the most positive effects on evergreen

newwww_df.columns

# A:
# lets try out spelling error ratio & news front page
temp_data = newwww_df[['label','spelling_errors_ratio','html_ratio']]
temp_data.describe()

temp_data['html_ratio'].plot(kind='hist')

temp_data['spelling_errors_ratio'].plot(kind='hist')

# EDA for front page news
temp_data['spelling_errors_ratio_binned'] = pd.qcut(temp_data['spelling_errors_ratio'], 5)
temp_data['html_ratio_binned'] = pd.qcut(temp_data['html_ratio'], 5)

fig, ax = plt.subplots(figsize=(20,5))
sns.factorplot(x='spelling_errors_ratio_binned', y='label', data=temp_data, ax=ax)

fig, ax = plt.subplots(figsize=(20,5))
sns.factorplot(x='html_ratio_binned', y='label', data=temp_data, ax=ax)

# spelling errors have quadratic function
# html ratio is linear

# convert both to ranks using stats.percentile
temp_data['spelling_errors_ratio_pctl'] = temp_data.spelling_errors_ratio.map(lambda x: stats.percentileofscore(temp_data.spelling_errors_ratio.values, x))
temp_data['html_ratio_pctl'] = temp_data.html_ratio.map(lambda x: stats.percentileofscore(temp_data.html_ratio.values, x))

# now we log individual models first
formula = 'label ~ spelling_errors_ratio_pctl'
model = sm.logit(formula, data=temp_data).fit()
model.summary()

formula = 'label ~ html_ratio_pctl'
model = sm.logit(formula, data=temp_data).fit()
model.summary()

# combine all with image ratio
temp_data['image_ratio_pctl'] = image_data['image_ratio_pctl']

formula = 'label ~ html_ratio_pctl + spelling_errors_ratio_pctl + image_ratio_pctl + np.power(image_ratio_pctl,2)'
model = sm.logit(formula, data=temp_data).fit()
model.summary()

model.params

# when
np.exp(model.params)

# for 1 unit increase in html ratio, the log odds of evergreen decrease by ~ 0.992076
# apply same protocol for all the parameters

title_data = newwww_df[['title','label']]

def title_len(x):
    try:
        return len(x.split())
    except:
        return 0.

# calculate the number of words in the title and plot distribution
title_data['title_words'] = title_data.title.map(title_len)
sns.distplot(title_data.title_words, bins=30, kde=False)

title_data['title_words_binned'] = pd.qcut(title_data['title_words'], 5)

sns.factorplot('title_words_binned', 'label', data=title_data, aspect=2).set_xticklabels(rotation=45, 
                                                                                 horizontalalignment='right')



