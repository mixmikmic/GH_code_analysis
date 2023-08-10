from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

cnn_model_1 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), name='Conv2D'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
])

cnn_model_2 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), name='Conv2D'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
])

for model in [cnn_model_1, cnn_model_2]: model.summary()



