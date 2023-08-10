get_ipython().system('pip3 -q install sklearn pandas seaborn')
get_ipython().system("pip3 -q install 'keras==2.0.6' --force-reinstall")

import pandas as pd
import sklearn
import keras
print("keras version {} installed".format(keras.__version__))
print("pandas version {} installed".format(pd.__version__))
print("scikit-learn version {} installed".format(sklearn.__version__))

# Import pandas 
import train_util as util

from importlib import reload
reload(util)

helper = util.LendingClubModelHelper()

# Read in lending club data 
helper.read_csv("lc-2015-loans.zip", 
                util.APPLICANT_NUMERIC +
                util.APPLICANT_CATEGORICAL +
                util.CREDIT_NUMERIC +
                util.LABEL)


print(helper.lcdata.info(null_counts = True, memory_usage = "deep", verbose = True))

get_ipython().magic('matplotlib inline')

import plot_util as plots

# Show a correlation matrix of the features in our data set
plots.plot_correlation_matrix(helper.lcdata)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.rcParams["figure.figsize"] = (20,8)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Create violinplot
plt.subplot(121)
v1 = sns.violinplot(x = "fico_range_low", y="grade", data=helper.lcdata)
v1.axes.set_title("FICO Low Range by Loan Grade", fontsize=20)

plt.subplot(122)
v2 = sns.violinplot(x = "acc_open_past_24mths", y="grade", data=helper.lcdata)
v2.axes.set_title("Accts Opened in Past 24 Months by Loan Grade", fontsize=20)

# Show the plot
plt.show()

melted = pd.melt(helper.lcdata, id_vars = ["grade"], value_vars=["annual_inc", "avg_cur_bal"])

# We unfortunately lose our categorical ordering with the pd.melt() call and need to add it back
melted["grade"] = melted["grade"].astype("category", categories=["A", "B", "C", "D", "E", "F", "G"], ordered=True)

sns.stripplot(x="grade", y="value", hue="variable", data=melted, jitter = True);

import os

# Divide the data set into training and test sets
helper.split_data(util.APPLICANT_NUMERIC + util.CREDIT_NUMERIC,
                  util.APPLICANT_CATEGORICAL,
                  util.LABEL,
                  test_size = 0.2,
                  row_limit = os.environ.get("sample"))

# Inspect our training data
print("x_train contains {} rows and {} features".format(helper.x_train.shape[0], helper.x_train.shape[1]))
print("y_train contains {} rows and {} features".format(helper.y_train.shape[0], helper.y_train.shape[1]))

print("x_test contains {} rows and {} features".format(helper.x_test.shape[0], helper.x_test.shape[1]))
print("y_test contains {} rows and {} features".format(helper.y_test.shape[0], helper.y_test.shape[1]))

# Loan grade has been one-hot encoded
print("Sample one-hot encoded 'y' value: \n{}".format(helper.y_train.sample()))

# %load model_definition.py
"""Create Keras model"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm

def create_model(input_dim, output_dim):
    # create model
    model = Sequential()
    # input layer
    model.add(Dense(100, input_dim=input_dim, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    # hidden layer
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    # output layer
    model.add(Dense(output_dim, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from model_definition import create_model

history = helper.train_model(create_model)

sns.reset_orig()

plots.plot_history(history)

import numpy as np
from sklearn.metrics import f1_score

y_pred = helper.model.predict(helper.x_test.as_matrix())

# Revert one-hot encoding to classes
y_pred_classes = pd.DataFrame((y_pred.argmax(1)[:,None] == np.arange(y_pred.shape[1])),
                              columns=helper.y_test.columns,
                              index=helper.y_test.index)

y_test_vals = helper.y_test.idxmax(1)
y_pred_vals = y_pred_classes.idxmax(1)

# F1 score
# Use idxmax() to convert back from one-hot encoding
f1 = f1_score(y_test_vals, y_pred_vals, average='weighted')
print("Test Set Accuracy: {:.2%}   (NOTE: Best results expected when training on the FULL dataset)".format(f1))


from sklearn.metrics import confusion_matrix

from importlib import reload
reload(plots)

# Confusion matrix
cfn_matrix = confusion_matrix(y_test_vals, y_pred_vals).astype(float)
plots.plot_confusion_matrix(cfn_matrix, [l for l in "ABCDEFG"])



