import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from IPython.core.pylabtools import figsize
figsize(10, 8)

# Step 1: create a dataframe  from slide 61
# results = 

roc = pd.DataFrame(index = results['threshold'], columns=['recall', 'precision', 'f1', 'tpr', 'fpr'])

for item_value in results.iterrows():
    
    #
    #
    #  STEP 2: Write your own code to calculate recall, precision, f1, tpr and fpr (15)
    #
    #
    pass

roc.reset_index()

figsize(10, 8)

plt.style.use('seaborn-dark-palette')

thresholds = [str(t) for t in results['threshold']]

# Step 3: Write a line of code to plot the roc curve (5)


# Step 4: add another line for naive classifier (5)


# Step 5: write threshold values at each point when drawing the curve (5)


plt.legend(prop={'size':14})
plt.ylabel('True Positive Rate', size = 16); plt.xlabel('False Positive Rate', size = 16);
plt.title('Receiver Operating Characteristic Curve', size = 20);

