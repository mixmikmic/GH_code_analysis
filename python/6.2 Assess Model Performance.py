from IPython.display import Image
Image("/Users/surthi/gitrepos/ml-notes/images/hold-out-cross-validation.jpg")

Image("/Users/surthi/gitrepos/ml-notes/images/k-fold-cross-validation.jpg")

get_ipython().magic("run '6.1 Model Evaluation and HyperParameter Tuning.ipynb'")

from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)

scores = []
for k, (train,test) in enumerate(kfold):
    p.fit(X_train[train], y_train[train])
    score = p.score(X_train[test], y_train[test])
    scores.append(p.score(X_train[test], y_train[test]))
    print('%d th iter, ClassDist: %s, AccScore: %.3f' %(k+1, np.bincount(y_train[train]), score))
    
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=p, X=X_train, y=y_train, cv=10, n_jobs=1)
print "Scores:", scores
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

