get_ipython().magic('matplotlib inline')

# numbers
import numpy as np
import pandas as pd

# plots
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns

# utils
import os, re
from pprint import pprint

# learn you some machines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

testing_df = pd.read_csv('data/optdigits/optdigits.tes',header=None)
X_testing,  y_testing  = testing_df.loc[:,0:63],  testing_df.loc[:,64]

training_df = pd.read_csv('data/optdigits/optdigits.tra',header=None)
X_training, y_training = training_df.loc[:,0:63], training_df.loc[:,64]

print X_training.shape
print y_training.shape

mat = X_training.loc[0,:].reshape(8,8)
print mat
plt.imshow(mat)
gca().grid(False)
plt.show()

def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    
    ## Automatic:
    #X_cov = np.cov(X_std.T)
    
    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    
    return X_std, X_mean, X_cov

X_std, X_mean, X_cov = get_normed_mean_cov(X_training)
X_std_validation, _, _ = get_normed_mean_cov(X_testing)

# Make PCA
pca2 = PCA(n_components=2, whiten=True)
pca2.fit(X_std)
X_red = pca2.transform(X_std)

# Make SVC
linclass2 = SVC()
linclass2.fit(X_red,y_training)

# To color each point by the digit it represents,
# create a color map with 10 elements (10 RGB values).
# Then, use the system response (y_training), which conveniently
# is a digit from 0 to 9.
def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return [ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(10)
colors = [colorz[yy] for yy in y_training]

scatter(X_red[:,0], X_red[:,1], 
        color=colors, marker='*')
xlabel("Principal Component 0")
ylabel("Principal Component 1")
title("Handwritten Digit Data in\nPrincipal Component Space",size=14)
show()

# Now see how good it is:
# Use the PCA to fit the validation data,
# then use the classifier to classify digits.
X_red_validation = pca2.transform(X_std_validation)
yhat_validation = linclass2.predict(X_red_validation)

y_validation = y_testing

pca2_cm = confusion_matrix(y_validation,yhat_validation)
sns.heatmap(pca2_cm, square=True, cmap='inferno')
title('Confusion Matrix:\n2-Component PCA + SVC')
ylabel('True')
xlabel('Predicted')
show()

total = pca2_cm.sum(axis=None)
correct = pca2_cm.diagonal().sum()
print "2-Component PCA Accuracy: %0.2f %%"%(100.0*correct/total)

# MSE associated with back-projection:
X_orig = X_std
X_hat = pca2.inverse_transform( pca2.transform(X_orig) )

mse = ((X_hat - X_orig)**2).mean(axis=None)
print mse

# Make PCA
pca5 = PCA(n_components=5, whiten=True)
pca5.fit(X_std)
X_red = pca5.transform(X_std)

# Make SVC
linclass5 = SVC()
linclass5.fit(X_red,y_training)

# Use the PCA to fit the validation data,
# then use the classifier to classify digits.
X_red_validation = pca5.transform(X_std_validation)
yhat_validation = linclass5.predict(X_red_validation)

y_validation = y_testing

pca5_cm = confusion_matrix(y_validation,yhat_validation)
sns.heatmap(pca5_cm, square=True, cmap='inferno')
title('Confusion Matrix:\n5-Component PCA + SVC')
show()

total = pca5_cm.sum(axis=None)
correct = pca5_cm.diagonal().sum()
print "5-Component PCA Accuracy: %0.2f %%"%(100.0*correct/total)

def pca_mse_accuracy(n):
    """
    Creates a PCA model with n components,
    reduces the dimensionality of the validation data set,
    then computes two error metrics:
    * Percent correct predictions using linear classifier
    * Back-projection error (MSE of back-projected X minus original X)
    """
    
    
    X_std, _, _ = get_normed_mean_cov(X_training)
    X_std_validation, _, _ = get_normed_mean_cov(X_testing)
    
    #########
    # Start by making PCA/SVC
    # Train on training data
    
    # Make PCA
    pca = PCA(n_components=n, whiten=True)
    pca.fit(X_std)
    X_red = pca.transform(X_std)

    # Make SVC
    linclass = SVC()
    linclass.fit(X_red,y_training)

    
    ########
    # Now transform validation data
    
    # Transform inputs and feed them to linear classifier
    y_validation = y_testing
    X_red_validation = pca.transform(X_std_validation)
    yhat_validation = linclass.predict(X_red_validation)
    
    # Evaluate confusion matrix for predictions
    cm = confusion_matrix(y_validation,yhat_validation)
    total = cm.sum(axis=None)
    correct = cm.diagonal().sum()
    accuracy = 1.0*correct/total
    
    
    # MSE associated with back-projection:
    X_orig = X_std
    X_hat = pca.inverse_transform( pca.transform(X_orig) )
    mse = ((X_hat - X_orig)**2).mean(axis=None)
    
    
    
    
    return mse, accuracy


mses = []
accuracies = []
N = 33
for i in range(1,N):
    m, a = pca_mse_accuracy(i)
    
    print "%d-component PCA: MSE = %0.4g, Accuracy = %0.2f"%(i,m,a*100.0)
    
    mses.append((i,m))
    accuracies.append((i,a))
    
mses = np.array(mses)
accuracies = np.array(accuracies)

# Now plot the accuracy
plt.plot(accuracies[:,0],accuracies[:,1],'bo-')
plt.title("Percent Correct: Accuracy of Predictions",size=14)
plt.xlabel("Number of Principal Components")
plt.ylabel("Percent Correct")
plt.show()

Nmax = 64

pcafull = PCA(n_components=Nmax, whiten=True)
_ = pcafull.fit(X_std)
explained_var_sum = pcafull.explained_variance_.sum()

explained_var = np.cumsum(pcafull.explained_variance_[:N])
explained_var /= explained_var_sum

plot(mses[:,0],mses[:,1],'-o',label='MSE')
plot(mses[:,0],1.0-mses[:,1],'-o',label='1.0 - MSE')
plot(range(1,len(explained_var)+1),explained_var,'-o',label='Expl Var')

xlabel('Number of Principal Components')
ylabel('')
title('Mean Square Error, Principal Components Analysis')

legend(loc='best')
show()

