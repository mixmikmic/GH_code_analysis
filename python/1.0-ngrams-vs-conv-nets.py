import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

df = pd.read_csv("../reports/accuracy_wrt_train_size.csv")
x = np.array(df["train_size"].tolist()) / 1000
y_cnn = df["cnn_acc"].tolist()
y_lr = df["ngrams_lr_acc"].tolist()

plt.figure(figsize=(15, 8))
plt.semilogx(x, y_cnn, marker="o")
plt.semilogx(x, y_lr, marker="o")
plt.xticks(x, x.astype(np.uint))

plt.legend(["ConvNet", "ngrams + Logistic Regression"])
plt.ylabel("Accuracy")
plt.xlabel("Train Samples x1000")

plt.show()

