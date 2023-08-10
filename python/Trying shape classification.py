import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix

data = pd.read_csv('data.csv')
print(data.describe())

no_of_classes = 36
label_indexes = {}
for i in range(26):
    if not chr(ord('a')+i) in ['h','j','v']: label_indexes[chr(ord('a')+i)] = i
for i in range(10): label_indexes[str(i)] = 26+i

x_axis = data['eccentricity']
y_axis = data['angle_90']
labels = data['label']

x, y =[], []

for i in range(no_of_classes):
    x.append([])
    y.append([])

for i in range(len(labels)):
    j = label_indexes[labels[i]]
    #j=labels[i]
    x[j].append(x_axis[i])
    y[j].append(y_axis[i])
        
fig = plt.figure()
ax1 = fig.add_subplot(111)
xs = np.arange(10)
ys = [i+xs+(i*xs)**2 for i in range(no_of_classes)]
colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

for i in range(1,no_of_classes+1):
    ax1.scatter(x[i-1], y[i-1], s=2, c=next(colors), label='%d'%(i))
plt.legend()
plt.xlabel('Eccentricity')
plt.ylabel('Angle')
plt.show()

x_axis = data['eccentricity']
y_axis = data['area_contour']
labels = data['label']

x, y =[], []

for i in range(no_of_classes):
    x.append([])
    y.append([])

for i in range(0,len(labels)):
    j = label_indexes[labels[i]]
    #j=labels[i]
    x[j].append(x_axis[i])
    y[j].append(y_axis[i])

fig = plt.figure()
ax1 = fig.add_subplot(111)
xs = np.arange(10)
ys = [i+xs+(i*xs)**2 for i in range(no_of_classes)]
colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

for i in range(no_of_classes):
    ax1.scatter(x[i], y[i], s=1, c=next(colors),label='%d'%(i))
plt.legend()
plt.xlabel('Eccentricity')
plt.ylabel('Area of contour')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
x_axis = data['proportion']
y_axis = data['eccentricity']
z_axis = data['r']
labels = data['label']

x, y, z = [], [], []

for i in range(no_of_classes):
    x.append([])
    y.append([])
    z.append([])

for i in range(len(labels)):
    j = label_indexes[labels[i]]
    #j=labels[i]
    x[j].append(x_axis[i])
    y[j].append(y_axis[i])
    z[j].append(z_axis[i])
        

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = np.arange(10)
ys = [i+xs+(i*xs)**2 for i in range(no_of_classes)]
colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

for i in range(1,no_of_classes+1):
    ax.scatter(x[i-1], y[i-1], z[i-1], s=i, c=next(colors), label='%d'%(i))
plt.show()

from sklearn.linear_model import perceptron
from numpy import array
import numpy as np
from random import randint

X = data[['angle_90', 'norm_r', 'scale',  'eccentricity', 'norm_area_contour']]
Y = data['label']

r_val = randint(1,1000)
print('Random state: %d'%(r_val))
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.3,random_state=204)
# 204
classifier = perceptron.Perceptron(n_iter=30, verbose=0)
classifier.fit(X_train,Y_train)

print('Confusion matrix:')
predictions = classifier.predict(X_test)
print(confusion_matrix(Y_test,predictions))
print('Accuracy on training set: %.3f' % (classifier.score(X_train,Y_train)*100))
print('Accuracy on testing set: %.3f' % (classifier.score(X_test,Y_test)*100))

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from random import randint

X = data[['angle_90', 'norm_r', 'scale',  'eccentricity', 'norm_area_contour']].values.tolist()
Y = data['label'].values.tolist()

r_val = randint(1,1000)
print('Random state: %d'%(r_val))
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.3,random_state=r_val)
# 100
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_train_labels = encoder.transform(Y_train)
dummy_train_labels = np_utils.to_categorical(encoded_train_labels)

encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_test_labels = encoder.transform(Y_test)
dummy_test_labels = np_utils.to_categorical(encoded_test_labels)

model = Sequential()
model.add(Dense(150, input_dim=5, activation='relu', name='h1'))
model.add(Dense(450, activation='relu', name='h2'))
model.add(Dense(650, activation='relu', name='h3'))
model.add(Dense(900, activation='relu', name='h4'))
model.add(Dense(33, activation='softmax', name='op'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,dummy_train_labels,epochs=20,verbose=1,validation_split=0.3)

score = model.evaluate(X_test,dummy_test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

from sklearn.neighbors import KNeighborsClassifier
from random import randint
from sklearn.model_selection import KFold

X = data[['angle_90', 'norm_r', 'proportion',  'eccentricity', 'norm_area_contour']].values
Y = data['label'].values

r_val = randint(1,1000)
print('Random state = %3d'%(r_val))
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.3, random_state=r_val)
# Letters -> 940,4
# with all data -> 254,618,912
# My data -> 985,323,802,218,258

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)

acc = classifier.score(X_test,Y_test)
print('Accuracy: %.3f' % (acc*100))
print('Approximate incorrect samples: %d/%d'%((1-acc)*len(X_test),len(X_test)))

from sklearn.ensemble import RandomForestClassifier

X = data[['angle_90', 'norm_r', 'proportion',  'eccentricity', 'norm_area_contour']].values
Y = data['label'].values

r_val = randint(1,1000)
print('Random state = %3d'%(r_val))
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.3, random_state=r_val)

classifier = RandomForestClassifier(n_estimators=100, max_depth=45, random_state=r_val, warm_start=True, max_features='log2')
classifier.fit(X_train, Y_train)

print('Accuracy: %.3f' % (classifier.score(X_test,Y_test)*100))

