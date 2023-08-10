# Import the libraries we will be using
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

# I've abstracted some of what we'll be doing today into a library.
# You can take a look at this code if you want by going into `dstools/data_tools.py`
from dstools import data_tools

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 14, 10

# Get some data
target_name, variable_names, data, Y = data_tools.create_data()

# Grab the predictors with sufficient complexity (3rd order interaction terms)
X = data_tools.X(complexity=3)

# Plot different regularization values for L1 and L2 regularization
for regularization in ['L2', 'L1']:
    # Get a table of the model coefficients
    coefs = pd.DataFrame(columns=['C'] + list(X.columns))
    
    # Print what we are doing
    print "Fitting with %s regularization:" % regularization
    position = 0
    
    # Try some regularization values
    for i in range(-6, 3):
        # Modeling
        c = np.power(10.0, i)
        model = LogisticRegression(penalty=regularization.lower(), C=c)
        model.fit(X, Y)
        
        # Plotting
        position += 1
        plt.subplot(3, 3, position)
        data_tools.Decision_Surface(X, Y, model)
        plt.title("C = " + str(np.power(10.0, i)))
        
        # Update coefficient table
        coefs.loc[i] = [c] + list(model.coef_[0])
    # Print and plot
    print coefs.to_string(index=False)
    plt.tight_layout()
    plt.show()

# Create hands-on data
from dstools import data_tools

X_handson, Y_handson = data_tools.handson_data()

# Try it out!

