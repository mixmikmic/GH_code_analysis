from create_lda_datasets import *
import scipy.sparse as sp
from classifiers import *

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_train, clf.predict(X_train)

    print("Detailed classification report:\n")
    print("Scores on training set.\n")
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Scores on test set.\n")
    print(classification_report(y_true, y_pred))

u = 37226353

X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(u)

X_train_lda, X_valid_lda, X_test_lda, y_train, y_valid, y_test = load_lda_dataset(u)

X_train_combined = sp.hstack((X_train, X_train_lda))
X_valid_combined = sp.hstack((X_valid, X_valid_lda))
X_test_combined = sp.hstack((X_test, X_test_lda))

X_train_lda

X_valid_lda

w1 = sum(y_train)/len(y_train)
w0 = 1 - w1
sample_weights = np.array([w0 if x==0 else w1 for x in y_train])

ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)

ds_sna = (X_train, X_valid, y_train, y_valid)

clf_sna = model_select_rdf((X_train, X_valid, y_train, y_valid))

clf_sna2 = model_select_sgd((X_train, X_valid, y_train, y_valid))

clf_sna3 = model_select_svc((X_train, X_valid, y_train, y_valid))

clf_comb = model_select_rdf((X_train_combined, X_valid_combined, y_train, y_valid))

clf_comb2 = model_select_sgd((X_train_combined, X_valid_combined, y_train, y_valid))

clf_comb3 = model_select_svc((X_train_combined, X_valid_combined, y_train, y_valid))



