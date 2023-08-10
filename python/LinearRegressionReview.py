import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"] = (13, 8)
plt.rcParams["font.size"] = 20

X = np.arange(-2.5, 2.5, 0.5)
X

m = 2
b = 1
Y = m * X + b
plt.plot(X, Y)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.grid(True)
plt.show()

x_data = np.random.rand(100).astype(np.float32)
plt.scatter(x_data, [0 for _ in range(100)])
plt.show()

y_data = 3 * x_data + 2
plt.scatter(x_data, y_data)
plt.grid(True)
plt.show()

y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.2))(y_data)
plt.scatter(x_data, y_data)
plt.show()

m = tf.Variable(1.0)
b = tf.Variable(0.2)
y = tf.multiply(m, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

train_data = []
for step in range(100):
    slope_and_intercept = session.run([train, m, b])[1:]
    if step % 5 == 0:
        print("Step: {} {}".format(step, slope_and_intercept))
        train_data.append(slope_and_intercept)

color_red, color_green, color_blue = 1.0, 1.0, 0.0  # The RGB values.
for slope_and_intercept in train_data:
    color_blue += 1.0 / len(train_data)
    color_green -= 1.0 / len(train_data)
    if color_blue > 1.0:
        color_blue = 1.0
    if color_green < 0.0:
        color_green = 0.0
    m, b = slope_and_intercept
    regression_line_y = np.vectorize(lambda x: m * x + b)(x_data)
    line = plt.plot(x_data, regression_line_y)
    plt.setp(line, color=(color_red, color_green, color_blue))
    
plt.plot(x_data, y_data, "ro")
plt.show()

