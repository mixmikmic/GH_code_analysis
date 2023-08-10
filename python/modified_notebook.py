import numpy as np
import pandas as pd
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df,test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# Create training and test matrix
R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    R[line[1]-1, line[2]-1] = line[3]  

T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))

lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space
m, n = R.shape # Number of users and items
n_epochs = 15 # Number of epochs

P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix

get_ipython().run_cell_magic('time', '', 'train_errors = []\ntest_errors = []\n\n# Repeat until convergence\nfor epoch in range(n_epochs):\n    # Fix Q and estimate P\n    for i, Ii in enumerate(I):\n        nui = np.count_nonzero(Ii) # Number of items user i has rated\n        if (nui == 0): nui = 1 # Be aware of zero counts!\n    \n        # Least squares solution\n        Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E\n        Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))\n        P[:,i] = np.linalg.solve(Ai,Vi)\n        \n    # Fix P and estimate Q\n    for j, Ij in enumerate(I.T):\n        nmj = np.count_nonzero(Ij) # Number of users that rated item j\n        if (nmj == 0): nmj = 1 # Be aware of zero counts!\n        \n        # Least squares solution\n        Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E\n        Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))\n        Q[:,j] = np.linalg.solve(Aj,Vj)\n    \n    train_rmse = rmse(I,R,Q,P)\n    test_rmse = rmse(I2,T,Q,P)\n    train_errors.append(train_rmse)\n    test_errors.append(test_rmse)\n    \n    print("[Epoch %d/%d] train error: %f, test error: %f" \\\n    %(epoch+1, n_epochs, train_rmse, test_rmse))\n    \nprint("Algorithm converged")')

# First, re-initialize P and Q
P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie

get_ipython().run_cell_magic('time', '', '\n# Uset different train and test errors arrays so I can plot both versions later\ntrain_errors_fast = []\ntest_errors_fast = []\n\n# Repeat until convergence\nfor epoch in range(n_epochs):\n    # Fix Q and estimate P\n    for i, Ii in enumerate(I):\n        nui = np.count_nonzero(Ii) # Number of items user i has rated\n        if (nui == 0): nui = 1 # Be aware of zero counts!\n    \n        # Least squares solution\n        \n        # Replaced lines\n        #-----------------------------------------------------------\n        # Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E\n        # Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))\n        #-----------------------------------------------------------\n        \n        # Added Lines\n        #-------------------------------------------------------------------\n        # Get array of nonzero indices in row Ii\n        Ii_nonzero = np.nonzero(Ii)[0]\n        # Select subset of Q associated with movies reviewed by user i\n        Q_Ii = Q[:, Ii_nonzero]\n        # Select subset of row R_i associated with movies reviewed by user i\n        R_Ii = R[i, Ii_nonzero]\n        Ai = np.dot(Q_Ii, Q_Ii.T) + lmbda * nui * E\n        Vi = np.dot(Q_Ii, R_Ii.T)\n        #-------------------------------------------------------------------\n        \n        P[:, i] = np.linalg.solve(Ai, Vi)\n        \n    # Fix P and estimate Q\n    for j, Ij in enumerate(I.T):\n        nmj = np.count_nonzero(Ij) # Number of users that rated item j\n        if (nmj == 0): nmj = 1 # Be aware of zero counts!\n        \n        # Least squares solution\n        \n        # Removed Lines\n        #-----------------------------------------------------------\n        # Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E\n        # Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))\n        #-----------------------------------------------------------\n        \n        # Added Lines\n        #-----------------------------------------------------------------------\n        # Get array of nonzero indices in row Ij\n        Ij_nonzero = np.nonzero(Ij)[0]\n        # Select subset of P associated with users who reviewed movie j\n        P_Ij = P[:, Ij_nonzero]\n        # Select subset of column R_j associated with users who reviewed movie j\n        R_Ij = R[Ij_nonzero, j]\n        Aj = np.dot(P_Ij, P_Ij.T) + lmbda * nmj * E\n        Vj = np.dot(P_Ij, R_Ij)\n        #-----------------------------------------------------------------------\n        \n        Q[:,j] = np.linalg.solve(Aj,Vj)\n    \n    train_rmse = rmse(I,R,Q,P)\n    test_rmse = rmse(I2,T,Q,P)\n    train_errors_fast.append(train_rmse)\n    test_errors_fast.append(test_rmse)\n    \n    print("[Epoch %d/%d] train error: %f, test error: %f" \\\n    %(epoch+1, n_epochs, train_rmse, test_rmse))\n    \nprint("Algorithm converged")')

# Check performance by plotting train and test errors
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data (Original)');
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data (Original)');
# Added curves for errors from updated algorithm to make sure the accuracy is unchanged (aside from random deviations)
plt.plot(range(n_epochs), train_errors_fast, marker='o', label='Training Data (Updated)');
plt.plot(range(n_epochs), test_errors_fast, marker='v', label='Test Data (Updated)');
plt.title('ALS-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# Calculate prediction matrix R_hat (low-rank approximation for R)
R_hat = pd.DataFrame(np.dot(P.T,Q))
R = pd.DataFrame(R)

# Compare true ratings of user 17 with predictions
ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']

ratings

predictions = R_hat.loc[16,R.loc[16,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
top5 = predictions.sort_values(ascending=False).head(n=5)
recommendations = pd.DataFrame(data=top5)
recommendations.columns = ['Predicted Rating']

recommendations

