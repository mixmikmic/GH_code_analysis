get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 2')

import sys
print(sys.version)

import sys
sys.path.append("../python")
import setup_dataset

data, labels = setup_dataset.setup_simple_iterables("with_dc")

X_train, X_test, y_train, y_test = setup_dataset.slice_data(data, labels)

# Setting up various complexities for the different algorithms.
# Number of neighbors
knn_c = (2, 4, 10, 50)
# Maximum depth in a decision tree
dtc_c = (2, 5, 10, 50)
# complexities for the rbf kernel
svc_c = (1, 1000, 1000000)
# Number of estimators in the random forest classifier
rfc_c = (1, 10, 100, 1000, 10000, 100000)
# Number of parallel jobs (CPU)
rfc_jobs = (3, -2)
gpc_jobs = (3, -2)
# Number of iteration in the Gaussian Process Classifier
gpc_c = (20, 50, 100)

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
scaler, X_train_scaled, X_test_scaled = setup_dataset.scale_sliced_data(X_train, X_test, StandardScaler())

knn_list, knn_accs, knn_pred, knn_pred_times, knn_fit_times = setup_dataset.run_knn(X_train, X_test, y_train, y_test, knn_c)

setup_dataset.compute_cm(y_test, knn_pred, knn_c)

knn_list_scaled, knn_accs_scaled, knn_pred_scaled, knn_pred_times_scaled, knn_fit_times_scaled =setup_dataset.run_knn(X_train_scaled, X_test_scaled, y_train, y_test, knn_c)

setup_dataset.compute_cm(y_test, knn_pred_scaled, knn_c)

for line in knn_accs :
    print(line)
print("====================") 
for line in knn_accs_scaled:
    print(line)

dtc_list, dtc_accs, dtc_pred, dtc_pred_times, dtc_fit_times = setup_dataset.run_decision_tree(X_train, X_test, y_train, y_test, dtc_c)

dtc_list_scaled, dtc_accs_scaled, dtc_pred_scaled, dtc_pred_times_scaled, dtc_fit_times_scaled = setup_dataset.run_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test, dtc_c)

setup_dataset.compute_cm(y_test, dtc_pred, dtc_c)

setup_dataset.compute_cm(y_test, dtc_pred_scaled, dtc_c)

for line in dtc_accs :
    print(line)
print("====================") 
for line in dtc_accs_scaled:
    print(line)

nbc_list, nbc_accs, nbc_pred, nbc_pred_times, nbc_fit_times = setup_dataset.run_naive_bayes(X_train, X_test, y_train, y_test, (1,))

nbc_list_scaled, nbc_accs_scaled, nbc_pred_scaled, nbc_pred_times_scaled, nbc_fit_times_scaled = setup_dataset.run_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test, (1,))

setup_dataset.compute_cm(y_test, nbc_pred, [1])

setup_dataset.compute_cm(y_test, nbc_pred_scaled, [1])

abc_list, abc_accs, abc_pred, abc_pred_times, abc_fit_times = setup_dataset.run_adaboost(X_train, X_test, y_train, y_test, (1,))

abc_list_scaled, abc_accs_scaled, abc_pred_scaled, abc_pred_times_scaled, abc_fit_times_scaled = setup_dataset.run_adaboost(X_train_scaled, X_test_scaled, y_train, y_test, (1,))

setup_dataset.compute_cm(y_test, abc_pred, [1])

setup_dataset.compute_cm(y_test, abc_pred_scaled, [1])

qda_list, qda_accs, qda_pred, qda_pred_times, qda_fit_times = setup_dataset.run_quadratic(X_train, X_test, y_train, y_test, (1,))

qda_list_scaled, qda_accs_scaled, qda_pred_scaled, qda_pred_times_scaled, qda_fit_times_scaled = setup_dataset.run_quadratic(X_train_scaled, X_test_scaled, y_train, y_test, (1,))

setup_dataset.compute_cm(y_test, qda_pred, [1])

setup_dataset.compute_cm(y_test, qda_pred_scaled, [1])

svc_list, svc_accs, svc_pred, svc_pred_times, svc_fit_times = setup_dataset.run_svc(X_train, X_test, y_train, y_test, svc_c)

svc_list_scaled, svc_accs_scaled, svc_pred_scaled, svc_pred_times_scaled, svc_fit_times_scaled = setup_dataset.run_svc(X_train_scaled, X_test_scaled, y_train, y_test, svc_c)

setup_dataset.compute_cm(y_test, svc_pred, svc_c)

setup_dataset.compute_cm(y_test, svc_pred_scaled, svc_c)

for line in svc_accs :
    print(line)
print("====================") 
for line in svc_accs_scaled:
    print(line)

for line in svc_accs :
    print(line)
print("====================") 
for line in svc_accs_scaled:
    print(line)

# THIS MAKES THE KERNEL CRASH!
rfc_list, rfc_accs, rfc_pred, rfc_pred_times, rfc_fit_times = setup_dataset.run_random_forest(X_train, X_test, y_train, y_test, rfc_c, rfc_jobs)

rfc_list_scaled, rfc_accs_scaled, rfc_pred_scaled, rfc_pred_times_scaled, rfc_fit_times_scaled = setup_dataset.run_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, rfc_c, rfc_jobs)

setup_dataset.compute_cm(y_test, rfc_pred, rfc_c)

setup_dataset.compute_cm(y_test, rfc_pred_scaled, rfc_c)

gpc_list, gpc_accs, gpc_pred, gpc_pred_times, gpc_fit_times = setup_dataset.run_gaussian(X_train, X_test, y_train, y_test, gpc_c, gpc_jobs)

gpc_list_scaled, gpc_accs_scaled, gpc_pred_scaled, gpc_pred_times_scaled, gpc_fit_times_scaled = setup_dataset.run_gaussian(X_train_scaled, X_test_scaled, y_train, y_test, gpc_c, rfc_jobs)

setup_dataset.compute_cm(y_test, gpc_pred, gpc_c)

setup_dataset.compute_cm(y_test, gpc_pred_scaled, gpc_c)

import numpy as np
import matplotlib.pyplot as plt
plt.figure()
x = np.arange(len(knn_accs[0]))
y = [[] for _ in range(len(knn_accs[0]))]
for i in range(len(knn_accs[0])):
    y[i] = knn_accs[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
    # plt.scatter(x, y[i], label="data {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
x = np.arange(len(knn_accs[0]))
y = [[] for _ in range(len(knn_accs[0]))]
width = 0.2
for i in range(len(knn_accs[0])):
    y[i] = knn_accs[i]
    plt.bar(x- 1.5*width + width*i, y[i], width, align='center', label="complexity {}".format(i), alpha=0.8)
    # plt.scatter(x, y[i], label="data {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
x = np.arange(len(knn_fit_times[0]))
y = [[] for _ in range(len(knn_fit_times[0]))]
for i in range(len(knn_fit_times[0])):
    y[i] = knn_fit_times[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
    # plt.scatter(x, y[i], label="data {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
x = np.arange(len(knn_accs_scaled[0]))
y = [[] for _ in range(len(knn_accs_scaled[0]))]
for i in range(len(knn_accs_scaled[0])):
    y[i] = knn_accs_scaled[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
    # plt.scatter(x, y[i], label="data {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
x = np.arange(len(svc_accs[0]))
y = [[] for _ in range(len(svc_accs[0]))]
for i in range(len(svc_accs[0])):
    y[i] = svc_accs[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
x = np.arange(len(svc_accs_scaled[0]))
y = [[] for _ in range(len(svc_accs_scaled[0]))]
for i in range(len(svc_accs_scaled[0])):
    y[i] = svc_accs_scaled[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()



plt.figure()
x = np.arange(len(dtc_accs_scaled[0]))
y = [[] for _ in range(len(dtc_accs_scaled[0]))]
for i in range(len(dtc_accs_scaled[0])):
    y[i] = dtc_accs_scaled[i]
    plt.plot(x, y[i], linestyle='-', label="complexity {}".format(i))
    # plt.scatter(x, y[i], label="data {}".format(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

for line in dtc_accs :
    print(line)
print("====================") 
for line in dtc_accs_scaled:
    print(line)

import pickle

pickle.dump(knn_list[3][1], open('../weights/knn_full_data_set_4_neighbors.sav', 'wb'))
pickle.dump(dtc_list[3][3], open('../weights/dtc_full_data_set_depth_50.sav', 'wb'))
pickle.dump(svc_list_scaled[3][2], open('../weights/svc_full_data_set_rbf_1e6.sav', 'wb'))

from sklearn.externals import joblib

joblib.dump(knn_list[3][1], '../weights/knn_full_data_set_4_neighbors.pkl', protocol=2)
joblib.dump(dtc_list[3][3], '../weights/dtc_full_data_set_depth_50.pkl', protocol=2)
joblib.dump(svc_list_scaled[3][2], '../weights/svc_full_data_set_rbf_1e6.pkl', protocol=2)

X_test_large = X_test[3]
X_test_scaled_large = X_test_scaled[3]
y_test_large = y_test[3]

knn_saved = pickle.load(open('../weights/knn_full_data_set_4_neighbors.sav', 'rb'))
dtc_saved = pickle.load(open('../weights/dtc_full_data_set_depth_50.sav', 'rb'))
svc_saved = pickle.load(open('../weights/svc_full_data_set_rbf_1e6.sav', 'rb'))

print("The score achieved with the saved model is:\n")
print("K-nearest Neighbors =", knn_saved.score(X_test_large, y_test_large))
print("Decision Tree =", dtc_saved.score(X_test_large,y_test_large))
print("Support Vector Machine =", svc_saved.score(X_test_scaled_large,y_test_large))

knn_saved = joblib.load('../weights/knn_full_data_set_4_neighbors.pkl')
dtc_saved = joblib.load('../weights/dtc_full_data_set_depth_50.pkl')
svc_saved = joblib.load('../weights/svc_full_data_set_rbf_1e6.pkl')

result = knn_saved.score(X_test_large,y_test_large)
print("The score achieved with the saved model is:\n")
print("K-nearest Neighbors =", knn_saved.score(X_test_large, y_test_large))
print("Decision Tree =", dtc_saved.score(X_test_large,y_test_large))
print("Support Vector Machine =", svc_saved.score(X_test_scaled_large,y_test_large))

print X_test_large[4000]
print len(X_test_large)

scaler_saved = joblib.dump(scaler, '../weights/scaler_saved.pkl')



