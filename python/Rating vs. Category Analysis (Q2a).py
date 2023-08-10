import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

app = pd.read_pickle('/Users/krystal/Desktop/app_clean.p')
app.head()

app = app.drop_duplicates()
app

app['overall reviews'] = map(lambda x: int(x) if x!='' else np.nan, app['overall reviews'])
app['overall rating'] = map(lambda x: float(x) if x!='' else np.nan, app['overall rating'])
app['current rating'] = map(lambda x: float(x) if x!='' else np.nan, app['current rating'])

categories = app['category'].unique()
categories

app['category'].value_counts()

num_review_by_category = app.groupby(by = ['category'])['overall reviews'].sum()
num_review_by_category = pd.DataFrame(num_review_by_category)
num_review_by_category.reset_index(level = 0, inplace = True)
num_review_by_category

from matplotlib import font_manager as fm
from matplotlib import cm
cs = cm.Set1(np.arange(25)/25.)
patches, texts, autotexts = plt.pie(num_review_by_category['overall reviews'], labels = num_review_by_category['category'], explode = np.repeat(0.1,24), colors = cs, autopct = '%1.1f%%', startangle = 140)
proptease = fm.FontProperties()
proptease.set_size(6)
plt.setp(texts, fontproperties=proptease)
plt.setp(autotexts, fontproperties=proptease)
plt.show()

for i in range(0, len(categories)):
    app_category = app.loc[app['category'] == categories[i]]
    app_category['overall reviews'].plot(kind = "hist", bins = 5)
    plt.xlabel('Number of Reviews')
    plt.title('Distribution of number of reviews in category %s'%(categories[i]))
    plt.show()

for i in range(0, len(categories)):
    app_category = app.loc[app['category'] == categories[i]]
    app_category['overall rating'].plot(kind = "hist", bins = 5)
    plt.xlabel('Overall Rating')
    plt.title('Distribution of overall rating in category %s'%(categories[i]))
    plt.show()



