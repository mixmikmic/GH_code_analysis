import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

grad = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
grad.head()

grad.admit.value_counts()

# Split my data into training and test.
indep_vars = ['gre', 'gpa', 'rank']
x_train, x_test, y_train, y_test = train_test_split(grad[indep_vars],
                                                   grad.admit,
                                                   test_size = 0.3,
                                                   random_state = 1234)
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

# I build the same logistic regression model as from part 01.
# Logit link is assumed by default
glm_logit = sm.GLM(y_train, 
                   x_train,
                   sm.families.Binomial()).fit()

# The fitted values are stored as probabilities.
# i.e., They are the probabilities of the response variable being a "success" (coded as 1).
yhat_train = glm_logit.fittedvalues
yhat_train.head()

# I let my threshold value be 1/2, which is typical.
# So, any probability above 1/2 is more likely to be a "success", and anything below 1/2 is more likely to be a "failure".
# If you had some known reason, you could choose something besides 1/2, but that's rare.
p_thresh = 0.5
yhat_train = (yhat_train > p_thresh) * 1
yhat_train.head()

# Training classification error
np.mean(yhat_train == y_train)

# Test classification error
yhat_test = glm_logit.predict(x_test)
yhat_test = pd.Series(yhat_test)
yhat_test = (yhat_test > p_thresh) * 1
np.mean(yhat_test == y_test) # Not bad at all!

iris = sns.load_dataset('iris')
iris.head()

# The class lables are the 'species'.  There are three.
iris.species.value_counts()

# Notice that these classes are VERY easy to separate.
sns.lmplot('sepal_length', 'sepal_width', 
           data = iris, 
           hue = 'species',
           fit_reg = False)

sns.lmplot('petal_length', 'petal_width', 
           data = iris, 
           hue = 'species',
           fit_reg = False)

# Training/Test split
# Purposefully not using all four variables
indep_vars = ['petal_length', 'petal_width']

x_train, x_test, y_train, y_test = train_test_split(iris[indep_vars],
                                                   iris.species,
                                                   test_size = 0.3,
                                                   random_state = 1234)

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

# Train the model
# multi_logit = sm.MNLogit(y_train, x_train).fit()
# Note that this is regularized.  Even though that's not what I want, the sci-kit learn
# multinomial regression is much less bug-prone than statsmodel's.
multi_logit = skl.LogisticRegression()
multi_logit.fit(x_train, y_train)

# Get training and test predictions (as probabilities)
fit_train = multi_logit.predict(x_train)
fit_test = multi_logit.predict(x_test)

pd.Series(fit_train).value_counts()

# Training error
print('Predicted correctly: ', np.sum(fit_train == y_train))
print('Total training obs: ', len(fit_train))
print('Training error: ', np.mean(fit_train == y_train))

# Testing error - nice!
print('Predicted correctly: ', np.sum(fit_test == y_test))
print('Total test obs: ', len(fit_test))
print('Test error: ', np.mean(fit_test == y_test))

