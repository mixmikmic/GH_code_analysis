from keras.datasets import boston_housing

# Download the data

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(test_targets)

# Prepare the data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

partial_train_data = train_data[:300]
partial_train_targets = train_targets[:300]
val_data = train_data[300:]
val_targets = train_targets[300:]

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(partial_train_data, partial_train_targets,
                    epochs=200, batch_size=1, verbose=0,
                    validation_data = (val_data, val_targets))

import matplotlib.pyplot as plt

print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']

time = range(1,len(loss)+1)

plt.plot(time, loss, 'b-', label = 'training')
plt.plot(time, val_loss, 'r-', label = 'validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

#plt.clf()
plt.plot(time, mae, 'b-', label = 'training')
plt.plot(time, val_mae, 'r-', label = 'validation')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae = smooth_curve(mae[10:])
smooth_val_mae = smooth_curve(val_mae[10:])

plt.plot(range(1, len(smooth_val_mae) + 1), smooth_val_mae)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(train_data, train_targets,
                    epochs=90, batch_size=15, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_mae_score

test_targets

model.predict(test_data)

x = test_targets
y = model.predict(test_data)

from numpy import arange

x0 = arange(0, 50)
y0 = x0

plt.plot(x, y, 'ro', label = 'Test data')
plt.plot(x0, y0, 'b-', label = 'Perfect fit')
plt.xlabel('Target value')
plt.ylabel('Predicted value')
plt.legend()
plt.grid()
plt.show()



