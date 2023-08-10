import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')

af = np.load("results/alif_final/no_alif_exp_1.npz")
print(af.keys())

svm_diff = af["svm_res"].item()["diff"]
print(len(svm_diff))
rc_diff = af["rc_res"].item()["diff"]
print(len(rc_diff))

def plot_confusion(mast_diff):
    diff_10 = []
    diff_20 = []
    diff_30 = []

    for diff in mast_diff:
        if diff.shape[0] == 10:
            diff_10.append(diff)
        elif diff.shape[0] == 20:
            diff_20.append(diff)
        elif diff.shape[0] == 30:
            diff_30.append(diff)
        else:
            print("WTF!")
            
    diff = np.array(diff_10)
    x_val = np.arange(10)*np.ones((30,10))*3
    sns.regplot(x=x_val.flatten(), y=diff.flatten(), x_estimator=np.mean)
    x2_val = np.arange(20)*np.ones((30,20))*3.0/2.0
    diff2 = np.array(diff_20)
    sns.regplot(x=x2_val.flatten(), y=diff2.flatten(), x_estimator=np.mean)
    x3_val = np.arange(30)*np.ones((30,30))
    diff3 = np.array(diff_30)
    sns.regplot(x=x3_val.flatten(), y=diff3.flatten(), x_estimator=np.mean)

plot_confusion(svm_diff)

plot_confusion(rc_diff)

plot_confusion(rc_diff)
plot_confusion(svm_diff)



