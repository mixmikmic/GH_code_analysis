import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
#to display all graphs inside Jupyter notebook itself
#If not using Jupyter notebook, then use plt.show() to display graph in another windows

df = pd.DataFrame(data=np.random.randint(1,100, (100,3)),
                 columns=['C1','C2','C3'])
# DataFrame with 100 rows * 3 columns

df.head() #Default is 5 rows

df['C1'].plot.area() #Area Graph

df['C1'].plot.bar() #Bar Graph

df['C1'].plot.barh() #Horizontal Bar Graph

df.plot.bar() #Display graph for all columns

df.plot.barh()

df['C3'].plot.line() #Line Graph

df.plot.box() #Box Plot
#Displays average value, maximum distribution line, minimum and maximum value

df['C2'].plot.kde() #Kernel Density

df.head()

df.plot.scatter(x='C1', y='C2')
#Scatter plot
#Horizontal axis is C1, Vertical axis is C2

df['C1'].plot.hist()
#Histogram

df['C1'].plot.hist(bins=5)
#Change bin size

