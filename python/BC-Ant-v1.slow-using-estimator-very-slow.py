import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gym

get_ipython().run_line_magic('matplotlib', 'inline')

sess = tf.InteractiveSession()

with open('./ant_train_test.pkl', 'rb') as inf:
    X_tv, y_tv, X_test, y_test = pickle.load(inf)

print(X_tv.shape, X_test.shape, y_tv.shape, y_test.shape)

feature_columns = [tf.feature_column.numeric_column("x", shape=[111])]

regressor = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    label_dimension=8,
#     model_dir="/tmp/iris_model"
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_tv},
    y=y_tv,
    batch_size=128,
    num_epochs=5,
    shuffle=True)

regressor.train(input_fn=train_input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)

regressor.evaluate(input_fn=test_input_fn)

regressor.evaluate(input_fn=train_input_fn)



def pred_action(model, obs):
    res = list(
        regressor.predict(
            tf.estimator.inputs.numpy_input_fn(
                x={'x': obs.reshape(1, -1)},
                shuffle=False)))
    return res[0]['predictions']

env = gym.make('Ant-v1')

obs = env.reset()
totalr = 0
done = False
max_timesteps = 600
for k in range(max_timesteps):
    if (k + 1) % 20 == 0:
        print(k + 1, end=',')
    action = pred_action(regressor, obs[None,:])
    obs, r, done, _ = env.step(action)
    totalr += r
    env.render()
env.render(close=True)
print()
print(totalr)
print(np.mean(totalr))



