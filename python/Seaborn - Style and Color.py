import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
tips = sns.load_dataset('tips')
tips.head()

# Default Style
sns.countplot(x='sex',data=tips)

# Sets the default Seaborn Style i.e. 'DarkGrid' 
sns.set()
sns.countplot(x='sex',data=tips)

# Setting the color style 'darkgrid'
sns.set_style('darkgrid')
sns.countplot(x='sex',data=tips)

sns.set_style('whitegrid')
sns.countplot(x='sex',data=tips)

# Remove Spines
sns.set_style('whitegrid')
sns.countplot(x='sex',data=tips)
sns.despine(left=True,bottom=True)# by default right and top are true



plt.figure(figsize=(12,3))
sns.set_style()
sns.countplot(x='sex',data=tips)

sns.set_context('notebook') # Set context = 'poster' for large prints
sns.countplot(x='sex',data=tips)


sns.set_style()
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='inferno')

