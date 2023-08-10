# Setup
get_ipython().magic('load_ext sql')
# %sql postgresql://gpdbchina@10.194.10.68:55000/madlib
get_ipython().magic('sql postgresql://fmcquillan@localhost:5432/madlib')
get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X_train_D = pd.DataFrame(X_train, columns=['x1', 'x2'])

# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-7, high=7, size=(40, 2))
X_outliers_D = pd.DataFrame(X_outliers, columns=['x1', 'x2'])

b = plt.scatter(X_train[:, 0], X_train[:, 1], c='blue')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.show()

# Build tables
get_ipython().magic('sql DROP TABLE IF EXISTS X_train_D CASCADE')
get_ipython().magic('sql PERSIST X_train_D')
get_ipython().magic('sql ALTER TABLE X_train_D add column X float[]')
get_ipython().magic('sql update X_train_D set X = array[x1, x2]::float[]')

get_ipython().magic('sql DROP TABLE IF EXISTS X_outliers_D CASCADE')
get_ipython().magic('sql PERSIST X_outliers_D')
get_ipython().magic('sql ALTER TABLE X_outliers_D add column X float[]')
get_ipython().magic('sql update X_outliers_D set X = array[x1, x2]::float[]')

get_ipython().run_cell_magic('sql', '', "-- Train the model\nDROP TABLE IF EXISTS svm_out1, svm_out1_summary, svm_out1_random CASCADE;\nSELECT madlib.svm_one_class(\n    'X_train_D',    -- source table\n    'svm_out1',     -- output table\n    'X',            -- features\n    'gaussian',     -- kernel\n    'gamma=1, n_components=55, random_state=3', \n    NULL,           -- grouping \n    'init_stepsize=0.1, lambda=10, max_iter=100, tolerance=0'  \n    );\nSELECT * FROM svm_out1; ")

# Prediction
# First for the training data
get_ipython().magic('sql drop table if exists y_pred_train;')
get_ipython().magic("sql SELECT madlib.svm_predict('svm_out1', 'X_train_D', 'index', 'y_pred_train');")
y_pred_train = get_ipython().magic('sql SELECT * from y_pred_train; ')

# Next for the outliers
get_ipython().magic('sql drop table if exists y_pred_outliers;')
get_ipython().magic("sql SELECT madlib.svm_predict('svm_out1', 'X_outliers_D', 'index', 'y_pred_outliers');")
y_pred_outliers = get_ipython().magic('sql SELECT * from y_pred_outliers; ')

get_ipython().magic('sql SELECT * FROM y_pred_outliers limit 20; -- Show the outliers')
#%sql SELECT * FROM y_pred_train limit 20; -- Show the training data

# Predict over the decision grid for plotting
# xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x1', 'x2'])

get_ipython().magic('sql DROP TABLE IF EXISTS grid_points CASCADE')
get_ipython().magic('sql PERSIST grid_points')
get_ipython().magic('sql ALTER TABLE grid_points add column X float[]')
get_ipython().magic('sql update grid_points set X = array[x1, x2]::float[]')

# Plot the decision grid
get_ipython().magic('sql drop table if exists Z_D;')
get_ipython().magic("sql SELECT madlib.svm_predict('svm_out1', 'grid_points', 'index', 'Z_D');")
Z_D = get_ipython().magic('sql SELECT decision_function from Z_D order by index')
Z = np.array(Z_D)
Z = Z.reshape(xx.shape)

# Orange is not novel, green is novel
plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
plt.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='green')
#plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7), cmap=plt.cm.Blues_r)
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='blue')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.show()



