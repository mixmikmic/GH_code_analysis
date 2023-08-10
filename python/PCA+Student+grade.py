get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split

student_data_mat   = pd.read_csv("student-mat.csv",delimiter=";")
student_data_por   = pd.read_csv("student-por.csv",delimiter=";")
student_data = pd.merge(student_data_mat,student_data_por,how="outer")
student_data.head()

col_str = student_data.columns[student_data.dtypes == object]

student_data = pd.get_dummies(student_data, columns = col_str, drop_first = True)
student_data.info()

print(student_data[["G1","G2","G3"]].corr())

# Since, G1,G2,G3 have very high correlation, we can drop G1,G2
student_data.drop(axis = 1,labels= ["G1","G2"])

label = student_data["G3"].values
predictors = student_data.drop(axis = 1,labels= ["G3"]).values

pca = PCA(n_components=len(student_data.columns)-1)
pca.fit(predictors)
variance_ratio = pca.explained_variance_ratio_

variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(variance_ratio_cum_sum)
plt.plot(variance_ratio_cum_sum)

#Looking at above plot I'm taking 10 variables
pca = PCA(n_components=10)
pca.fit(predictors)
Transformed_vector =pca.fit_transform(predictors)
print(Transformed_vector)

lr_pca = linear_model.LinearRegression()

score_lr_pca = cross_val_score(lr_pca, Transformed_vector, label, cv=5)
print("PCA Model Cross Validation score : " + str(score_lr_pca))
print("PCA Model Cross Validation Mean score : " + str(score_lr_pca.mean()))

lr = linear_model.LinearRegression()
score_lr = cross_val_score(lr, predictors, label, cv=5)
print("LR Model Cross Validation score : " + str(score_lr))
print("LR Model Cross Validation Mean score : " + str(score_lr.mean()))



