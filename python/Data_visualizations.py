# Begin by importing pandas and seaborn, a data analysis tookit and graphing library, respectively. 

import pandas as pd

# To ignore warnings, use the following code to make the display more attractive.
# Import seaborn and matplotlib.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors, datasets
import numpy as np

sns.set(style="white", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

# To import the Iris dataset:
iris = datasets.load_iris() 

data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['Species'])
data['Species'] = data['Species'].map({0.0: 'setosa',  1.0: 'versicolor', 
                                      2.0: 'virginic'})

# 
iris.data

# STEP 1: View first 6 rows in the data

data.head(6)

# STEP 2: Get the frequency of each class of class labels(Species)

data.groupby('Species').count()

data.info()

# STEP 3: Change Dataframe column names to:
#  SepalLengthCm, SepalWifthCm, PetalLengthCm, PetalWidthCm, Species
    
data.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm',
                'PetalWidthCm','Species']

data.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", color="blue", s=50)
plt.show()

# using sns to create a graph by assigning each species an individual color.
import seaborn as sns
KS = {'color': ['blue', 'red', 'green']}
sns.FacetGrid(data, hue_kws=KS, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()
plt.show()

sns.boxplot(x="Species", y="PetalLengthCm", data=data )
plt.show()

ax= sns.boxplot(x="Species", y="PetalLengthCm", data=data)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=data, 
                  jitter=True, edgecolor="blue")
plt.show()



#Use pairplot to analyze the relationship between species for all characteristic combinations. 
# An observable trend shows a close relationship between two of the species

sns.pairplot(data, hue="Species", size=3)
plt.show()



# To make a Pandas boxplot grouped by species, use .boxplot
# Modify the figsize, by placing a value in the X and Y cordinates
data.boxplot(by="Species", figsize=(10, 10))
plt.show()



