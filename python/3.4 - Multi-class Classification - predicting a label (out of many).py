from sklearn.datasets import load_digits

digits = load_digits()

print(digits.DESCR)

len(digits.images)

digits.images[50]

digits.images[50].shape

digits.data[50]

digits.data[50].shape

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.gray()
plt.matshow(digits.images[50]) 

digits.target[50]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

from sklearn.svm import LinearSVC

classifier = LinearSVC(random_state=0)  # one-vs-rest by default
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

from sklearn.multiclass import OneVsOneClassifier

classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

print(classification_report(Y_test, Y_pred))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, shuffle=True, random_state=0)

cross_val_score(classifier, digits.data, digits.target, cv=cv, scoring='f1_macro').mean()

cross_val_score(classifier, digits.data, digits.target, cv=cv, scoring='f1_micro').mean()



