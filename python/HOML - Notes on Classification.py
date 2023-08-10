from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist

import numpy as np
np.random.seed(42) #for reproducting code!

X, y = mnist['data'], mnist['target']
X.shape

y.shape

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
print(some_digit_image[0:5])

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
          interpolation="nearest")
plt.axis("off")
plt.show()

#Neat, its a five!!!!! This is what the label says too.
print("Label: ",y[36000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

y_train_5

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3) #for reproducible results
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, scoring="accuracy", cv=3)

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X),1),dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, )

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

#So you see, false negative and false positive are both zero!

from sklearn.metrics import precision_score, recall_score
print("Precision Score: ", precision_score(y_train_5, y_train_pred))
print("Recall Score: ", recall_score(y_train_5, y_train_pred))

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])
y_scores

y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
y_scores[0:5]

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
thresholds[0:5]

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(recalls[:-1], precisions[:-1],"b--", label="Precision")
    plt.plot([1,0],[0,1],"y--") #not sure if this is good to do for this graph, it looks cooL!
    plt.axis([0,1,0,1])
    plt.xlabel("Recall")
    plt.title("Precision/Recall Tradeoff Plot")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

from sklearn.metrics import roc_curve

fpr,tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr,tpr,lw, label=None):
    plt.plot(fpr,tpr,linewidth=lw,label=label)
    plt.plot([0,1],[0,1],'o--') #r-- creates an orange line. Neat!
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr,lw=2)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores)

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                   method='predict_proba')

y_predicts_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)

print(y_probas_forest[0:5])
print(y_predicts_forest[0:5])

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", linewidth=4,label="SGD")
plot_roc_curve(fpr_forest, tpr_forest,lw=4,label="Random Forest")
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_score, recall_score
print("ROC Score: ", roc_auc_score(y_train_5, y_scores_forest))
print("Precision Score: ", precision_score(y_train_5, y_predicts_forest))
print("Recall Score: ", recall_score(y_train_5, y_predicts_forest))

sgd_clf.fit(X_train,y_train)
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

sgd_clf.classes_

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
ovo_clf.fit(X_train, y_train)
print('Prediction on some digit: ',ovo_clf.predict([some_digit]))
print('Num. Estimators: ',len(ovo_clf.estimators_))

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

forest_clf.predict_proba([some_digit])

print("RF scores: ", cross_val_score(forest_clf,X_train,y_train, cv=3,scoring='accuracy'))
print("SGD scores: ", cross_val_score(sgd_clf,X_train,y_train, cv=3, scoring='accuracy'))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_train_scaled[0:5]

print("RF scores: ", cross_val_score(forest_clf,X_train_scaled,y_train, cv=3,scoring='accuracy'))
print("SGD scores: ", cross_val_score(sgd_clf,X_train_scaled,y_train, cv=3, scoring='accuracy'))

from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
print('First ten: ',y_train_pred[0:10])

conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()

# step 1
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
print(np.around(norm_conf_mx, decimals=3))

# step 2
np.fill_diagonal(norm_conf_mx,0) #fill diagonal with zeros to highlight other differences
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# set up plot_digit and plot_digits functions

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))

plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.title('Correct: Classified as 3')
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.title('Incorrect: Classified as 5')
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.title('Incorrect: Classified 3s')
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.title('Correct: Classified as 5s')
plt.show()

#Note, not sure if I got these titles right!

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd] #create multiclass matrix
y_multilabel

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

#y_train_knn_pred = cross_val_predict(knn_clf, X_train,y_multilabel, cv=3)
#f1_score(y_multilabel, y_train_knn_pred, average='macro')

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

# Now we set the y variable to the original observation! 
y_train_mod = X_train
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

