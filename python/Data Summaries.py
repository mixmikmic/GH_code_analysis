import pandas as pd

# Sample code number: id number
#     Clump Thickness: 1 - 10
# 3. Uniformity of Cell Size: 1 - 10
# 4. Uniformity of Cell Shape: 1 - 10
# 5. Marginal Adhesion: 1 - 10
# 6. Single Epithelial Cell Size: 1 - 10
# 7. Bare Nuclei: 1 - 10
# 8. Bland Chromatin: 1 - 10
# 9. Normal Nucleoli: 1 - 10
# 10. Mitoses: 1 - 10
# 11. Class: (2 for benign, 4 for malignant)

    
names = ['sampleid', 'clumpthickness', 'sizeuniformity', 'shapeunformity', 
         'adhesion', 'epithelialsize', 'barenuclei', 'blandchromatin', 'normalnucleoli', 
         'mitoses', 'cellclass'] 

df = pd.read_csv('./breast-cancer-wisconsin.data', names=names)
# df.drop('sampleid')
df.drop('sampleid', axis=1, inplace=True)
df.head(10)

df.cellclass = (df.cellclass == 4).astype(int)

# It turns out one column is a string, but should be an int... 
df.barenuclei = df.barenuclei.values.astype(int)


df.describe()

# Check the class balance.  Turns out to be pretty good so we should have a relatively unbiased view
print 'Num Benign', (df.cellclass==2).sum(), 'Num Malignant', (df.cellclass==4).sum()

from pandas.tools.plotting import scatter_matrix
_ = scatter_matrix(df, figsize=(14,14), alpha=.4)



from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import svm

LR = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, 
                   fit_intercept=True, intercept_scaling=1, 
                   class_weight=None, random_state=None, 
                   solver='liblinear', max_iter=100, 
                   multi_class='ovr', verbose=1, 
                   warm_start=False, n_jobs=1)

X, Y = df.astype(np.float32).get_values()[:,:-1], df.get_values()[:,-1]

X2 = np.append(X,X**2, axis=1)
print X2.shape

LR.fit(X, Y)
print LR.score(X,Y)

C_list = np.logspace(-1, 2, 15)
CV_scores = []
CV_scores2 = [] 
for c in C_list: 
    LR = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c, 
                   fit_intercept=True, intercept_scaling=1, 
                   class_weight=None, random_state=None, 
                   solver='liblinear', max_iter=100, 
                   multi_class='ovr', verbose=1, 
                   warm_start=False, n_jobs=1)
    CV_scores.append(np.average(cross_validation.cross_val_score(LR, X, Y, cv=6, n_jobs=12)))
    
    svm_class = svm.SVC(C=c, kernel='linear', gamma='auto', coef0=0.0, 
        shrinking=True, probability=False, tol=0.001, cache_size=200, 
        class_weight=None, verbose=False, 
        max_iter=-1, decision_function_shape=None, random_state=None)
    CV_scores2.append(np.average(cross_validation.cross_val_score(svm_class, X, Y, cv=6, n_jobs=12)))
    
    

plt.plot(C_list, CV_scores, marker='o', label='Logistic Regression L1 loss')
plt.plot(C_list, CV_scores2, marker='o', label='SVM-Linear')
plt.xscale('log')
plt.xlabel(r'C = 1/$\lambda$')
plt.legend(loc=4)


from sklearn.metrics import confusion_matrix

LR = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1e10, 
               fit_intercept=True, intercept_scaling=1, 
               class_weight=None, random_state=None, 
               solver='liblinear', max_iter=100, 
               multi_class='ovr', verbose=1, 
               warm_start=False, n_jobs=1)
LR.fit(X[:300],Y[:300])

svm_class = svm.SVC(C=10., kernel='linear', gamma='auto', coef0=0.0, 
        shrinking=True, probability=True, tol=0.001, cache_size=200, 
        class_weight=None, verbose=False, 
        max_iter=-1, decision_function_shape=None, random_state=None)
svm_class.fit(X[:300],Y[:300])

# Confusion matrix
print 
print 'Confusion Matrix - LASSO Regression'
print confusion_matrix(y_true=Y[300:], y_pred=LR.predict(X[300:]))
print 'Confusion Matrix - SVM-Linear'
print confusion_matrix(y_true=Y[300:], y_pred=svm_class.predict(X[300:]))



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


plt.figure(figsize=(7,2))
plt.subplot(121)
prec, rec, thresh = precision_recall_curve(y_true=Y[300:], probas_pred=LR.predict_proba(X[300:])[:,1])
plt.plot(rec, prec,)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0,1)
plt.ylim(0,1)

plt.subplot(122)
fp, tp, thresh = roc_curve(y_true=Y[300:], y_score=LR.predict_proba(X[300:])[:,1])
AUC = roc_auc_score(y_true=Y[300:], y_score=LR.predict_proba(X[300:])[:,1])
roc_curve(y_true=Y[300:], y_score=LR.predict_proba(X[300:])[:,1])
plt.text(.05, .05, 'AUC=%1.3f'%AUC)
plt.plot(fp, tp, linewidth=2)
plt.xlabel('False Positives')
plt.ylabel('True Positives')





