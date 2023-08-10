# All required libraries are imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Label is droped from training data
df = pd.read_csv('train.csv', header=0)
df_label = df['label']
df = df.drop('label', axis=1)
df.head()

# The data is ampped to image to check whether the data and label is in sync with each other
r = np.random.randint(len(df.index))
print(r)
imgshow = plt.imshow(df.iloc[r].values.reshape(28,28), cmap='gray')
plt.title(df_label[r], fontsize=30)
plt.grid()
plt.show()

# K Nearest Neighbour Agorithm is implemented
knn = KNeighborsClassifier()

# Data is splitted randomly into training and validation data
df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.33, random_state=42)

# The classifier is treined with the training data
knn.fit(df_train.values, label_train.values)

# The trained classifier is used to predict values from validation data
prediction = knn.predict(df_test.values)

from sklearn.metrics import accuracy_score

# Accuracy score of prediction on validation data
accuracy_score(prediction,label_test.values)

# Test dataset is imported
df_test2 = pd.read_csv('test.csv', header=0)
df_test2.head()

# Test data is predicted
prediction2 = knn.predict(df_test2.values)

df_output = pd.DataFrame({
    'ImageId' : df_test2.index.values+1,
    'label': prediction2
})

# the predicted value is stored in a csv file
df_output.to_csv("./output/output.csv", sep=',', index=False)

