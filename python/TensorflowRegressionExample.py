import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic('matplotlib inline')

# Create Dataset
x_data = np.linspace(0.0,10.0,1000000)

x_data[:20]

noise = np.random.randn(len(x_data))

noise[:20]

# y = mX + b + noise
y_true = (0.5 * x_data) + 5 + noise

X_data = pd.DataFrame(data=x_data,columns=['X'])

y_data = pd.DataFrame(data=y_true,columns=['y'])

X_data.head()

y_data.head()

# Concatinate X and y along columns to make a single dataframe
df = pd.concat([X_data,y_data],axis=1)

df.head()

# Get 250 random sample values from dataframe and plot them
df.sample(n=250).plot(kind='scatter',x='X',y='y')

# Define Batch Size for Data to be Input
# Here, batch_size = 10 i.e. each batch will have 10 (X,y) values
batch_size = 10

np.random.randn(2)

# Weight
W = tf.Variable(1.580)

# Bias
b = tf.Variable(-0.050)

# Define Placeholders
# y = Wx + b
# Since, x takes in the value during run time, it is defined as a placeholder
# tf.placeholder(dtype, shape, name=None)
# Since, we'll be having batch size of 10 i.e. 10 samples per batch, x will have 10 values only. So, shape = [10].
x = tf.placeholder(tf.float32,shape=([batch_size]))

# Since, y will also have 10 values/labels due to batch_size = 10, so shape = [10].
y = tf.placeholder(tf.float32,shape=([batch_size]))

# y_pred
y_pred = W * x + b

# Loss Function
# error += (y - y_pred)**2
# tf.reduce_sum(): Computes the sum of elements across dimensions of a tensor.
error = tf.reduce_sum(tf.square(y - y_pred))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# Train
train = optimizer.minimize(error)

# Initialize all Variables
init = tf.global_variables_initializer()

# Run the Training inside the session
with tf.Session() as sess:
    sess.run(init)
    
    # Number of total batches
    # Batches = 1000
    # Each batch of size = 8 samples
    # Total samples = 8000
    batches = 1000
    
    for i in range(batches):
        # Get random Index from x_data equal to batch_size in number
        # ex. here it returns 10 random indexes within range "0 to 1000000".
        rand_idx = np.random.randint(len(x_data),size=batch_size)
        
        # Run the Gradient Descebt Optimizer to Reduce the Error 
        # x_data: Features, y_true: Labels ; Supervised Learning
        result,err = sess.run([train,error],feed_dict={x:x_data[rand_idx], y:y_true[rand_idx]})
       
        # Print Error at every Epoch
        if (i%100) == True:
            print('Epoch: {0} , Error: {1}'.format(i-1,err))
    
    print('Epoch: {0} , Error: {1}'.format(1000,err))
        
    # Get new values of Weight & Bias
    model_W, model_b = sess.run([W,b])

model_W

model_b

y_hat = model_W * x_data + model_b

y_hat

# Plot the Regression Line
df.sample(n=250).plot(kind='scatter',x='X',y='y')
plt.plot(x_data,y_hat,'r')

# Define a List of Feature Columns
feat_cols = [tf.feature_column.numeric_column(key='x',shape=[1])]

# Create the Estimator Model
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# Train Test Split using Scikit Learn
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x_data,y_true,test_size = 0.3, random_state=101)

X_train.shape

X_val.shape

# Set up Estimator Inputs
input_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

# Set up Estimator Training Inputs
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)

# Set up Estimator Test Inputs
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': X_val}, y_val, batch_size=8, num_epochs=1000, shuffle=False)

# Train the Estimator
estimator.train(input_fn=input_func, steps=1000)

# Get Evaluation Metrics
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

# Check the Performance on Test Data
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

print('Training Data Metrics: \n')
print(train_metrics)

print('Eval Metrics: \n')
print(eval_metrics)

# Get Predicitions
# New Data model has never seen
brand_new_data = np.linspace(0,10,10)

input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

list(estimator.predict(input_fn=input_fn_predict))

# Plot Predicted Values
predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

predictions

df.sample(n=250).plot(kind='scatter', x='X',y='y')
plt.plot(brand_new_data,predictions,'r*')

