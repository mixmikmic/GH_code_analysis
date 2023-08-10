from sklearn.datasets import load_iris

iris = load_iris()

X = iris['data']

y = iris['target']

y

y.dtype

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)

import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":np.array(X)},
      y=np.array(y),
      num_epochs=None,
      shuffle=True)

classifier.train(input_fn=train_input_fn, steps=20000)

predict_input_fn  = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(X_test)},
  num_epochs=1,
  shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [np.asscalar(p["classes"]) for p in predictions]
predicted_classes = [int(i) for i in predicted_classes]

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,predicted_classes))

score = accuracy_score(y_test, predicted_classes)
print('Accuracy: {0:f}'.format(score))

