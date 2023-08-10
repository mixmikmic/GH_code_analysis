# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')

# Load data
mov = pd.read_csv('Movie_gross_percentage.csv', encoding = 'latin1')

mov.head()

mov.describe()

mov.info()

# Visualization 1
vis1 = sns.factorplot(data = mov, x = 'Day of Week', kind = 'count', size = 10)

# Unique studios
mov.Studio.unique()

# Number of unique studios
len(mov.Studio.unique())

# Unique genres
mov.Genre.unique()

# Number of unique genres
len(mov.Genre.unique())

# Filter for genres
genre_filters = ['action','adventure','animiation','comedy','drama']
mov2 = mov[mov.Genre.isin(genre_filters)]

mov2.Genre.unique()

# Filter for genres made studios we're interested in
# mov3 = mov[(mov.Studio=='Fox') | (mov.Studio =='WB')]

studio_filters = ['Buena Vista Studios','Fox', 'Paramount Pictures','Sony','Universal','WB']
mov3 = mov2[mov2.Studio.isin(studio_filters)]

# Unique studios
print(mov3.Studio.unique())

# Number of unique studios
len(mov3)

# Create box plot
sns.set(style = 'darkgrid', palette = 'muted', color_codes = True)
ax = sns.boxplot(data = mov3, x = 'Genre', y ='Gross % US', 
            orient = 'v',color = 'lightgray')

plt.setp(ax.artists, alpha = 0.5)

sns.stripplot(data = mov3, x = 'Genre', y = 'Gross % US',
             jitter = True, size = 6, linewidth = 0, hue = 'Studio')

# Format title 
ax.axes.set_title('Domestic Gross % by Genre', fontsize = 30)

# Format x and y axis
ax.set_xlabel('Genre', fontsize = 20)
ax.set_ylabel('Gross % US', fontsize = 20)

# Format legend
ax.legend(bbox_to_anchor = (1.05, 1), loc = 2)
plt.show()

