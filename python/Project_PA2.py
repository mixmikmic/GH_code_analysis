import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn import metrics
get_ipython().magic('matplotlib inline')

from numpy.random import randn

from scipy import stats
import seaborn as sns

df = pd.read_csv('listings.csv')

df.head(2)

df['price'] = df['price'].replace('[\$,)]','',          regex=True).replace('[(]','-', regex=True).astype(float)
df.price.head(5)

reviews_column = ['price','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month']

scatter_plots = pd.scatter_matrix(df[reviews_column],figsize=(20,20))

#function for plotting scatter plots
def plot_scatter(x,y, title, x_label, y_label, face, axes):
    
    axes.scatter(x,y,color=face,alpha=0.5)

    
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    
    
    
    return axes
    
#plot histograms for each marker and each demographics
#in the following, instead of adding one subplot to a 4x2 grid at a time
#I can get all the subplot axes for the grid in one line 
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
ax1 = plot_scatter(df.number_of_reviews,df.price,
                'Number of reviews vs price', 
                'Number of reviews', 'Price',
                'green', 
                ax1)

ax2 = plot_scatter(df.review_scores_rating,df.price,
                'Review score rating vs Price', 
                'Review score rating', 'Price',
                'orange', 
                ax2)

ax3 = plot_scatter(df.review_scores_value,df.price,
                'Review score value vs Price', 
                'Review score value', 'Price', 
                'green', 
                ax3)

ax4 = plot_scatter(df.reviews_per_month,df.price,
                'Reviews per month vs Price', 
                'Reviews per month', 'Price',
                'orange', 
                ax4)

ax5 = plot_scatter(df.price,df.number_of_reviews,
                'Price vs number of Reviews', 
                'Price','Number of Reviews',
                'orange', ax5)



plt.tight_layout()
plt.show()

price_review = df[['number_of_reviews', 'price']].sort_values(by = 'price')

price_review.plot(x = 'price', y = 'number_of_reviews', figsize =(10,10), kind = 'area', title = 'Reviews vs Price', xlim=0,ylim=0)

df.cancellation_policy.value_counts(0)

from collections import Counter
from matplotlib import cm
cancel_type = df.cancellation_policy.value_counts(0)
cancel_type = cancel_type.drop(cancel_type["super_strict_30":"long_term"].index)
a=np.random.random(40)
cs=cm.Set1(np.arange(40)/40.)
cancel_type.plot.pie(colors=cs,
                   figsize=(10,10), 
                   autopct = '%.2f',
                   title = "Cancellation Policy Pie chart")

type(cancel_type)

cancel_type = cancel_type.drop(cancel_type["super_strict_30":"long_term"].index)

cancel_type



