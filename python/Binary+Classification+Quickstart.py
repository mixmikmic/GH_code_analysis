from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

bc = load_breast_cancer()

bc_df = pd.DataFrame(data = np.c_[bc['data'], bc['target']],
                     columns = bc['feature_names'].tolist() + ['target'])

bc_df.head()

from sklearn.model_selection import train_test_split

bc_train, bc_test = train_test_split(bc_df, test_size=0.2)

print("# of rows in training set = ",bc_train.size)
print("# of rows in test set = ",bc_test.size)

from microsoftml import rx_fast_linear

features = bc_df.columns.drop(["target"])
model = rx_fast_linear("target ~ " + "+".join(features), data=bc_train)

from microsoftml import rx_predict

prediction = rx_predict(model, data=bc_test)

prediction.head()

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(bc_test["target"], prediction["PredictedLabel"])

roc_auc = auc(fpr, tpr)
print(roc_auc)

import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Breast Cancer Prediction')
plt.legend(loc="lower right")
plt.show()

