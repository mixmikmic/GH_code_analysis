# Load libraries for Machine Learning
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

data = load_wine()

type(data)

X = data.data
y = data.target

# The features and labels are numpy arrays
type(X)
type(y)

# Verify the contents of the X (Features) and y (Labels) arrays have 178 samples each, and the X array has dimensionality of 13
X.shape

y.shape

X_train, X_test, y_train, y_test = train_test_split(wine, y,
                                                    test_size=0.30,
                                                    random_state=101)

# View the data before scaling
X_train

# Create an instance of the Scaler Class
scaler = StandardScaler()

# Note, fit will scale the data and transform will transform the data back into a numpy_array.
# Since the scale has already been fitted with the X_train data, for X_test we only need to do a transform.
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# View the data after it has been scaled
X_train

# Train a model (using 5 trees)
model = RandomForestClassifier( n_estimators = 5, criterion = 'entropy', random_state = 101 )
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# View the predicted values
y_pred

# View the actual values
y_test

# Create a confusion matrix with the actual and predicted values
cm = confusion_matrix(y_test, y_pred)

# View results (52 correct, 2 incorrect)
cm

# Calculate our accuracy
accuracy = 52 / 54
accuracy



