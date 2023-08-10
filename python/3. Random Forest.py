from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',n_estimators=1)

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

