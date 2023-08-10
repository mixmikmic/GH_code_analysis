# First, let's verify that the ATK client libraries are installed
import trustedanalytics as ta
print "ATK installation path = %s" % (ta.__path__)

# Next, look-up your ATK server URI from the TAP Console and enter the information below.
# This setting will be needed in every ATK notebook so that the client knows what server to communicate with.

# E.g. ta.server.uri = 'demo-atk-c07d8047.demotrustedanalytics.com'
ta.server.uri = 'ENTER URI HERE'

# This notebook assumes you have already created a credentials file.
# Enter the path here to connect to ATK
ta.connect('myuser-cred.creds')

# Create a new frame by uploading rows
frame = ta.Frame(ta.UploadRows([[4.9,1.4,0], 
                                [4.7,1.3,0], 
                                [4.6,1.5,0], 
                                [6.3,4.9,1],
                                [6.1,4.7,1], 
                                [6.4,4.3,1], 
                                [6.6,4.4,1], 
                                [7.2,6.0,2],
                                [7.2,5.8,2], 
                                [7.4,6.1,2], 
                                [7.9,6.4,2]],
                                 [('Sepal_Length', ta.float64),('Petal_Length', ta.float64), ('Class', int)]))

# Consider the following frame containing three columns.
frame.inspect()

# Create a new model and train it
model = ta.LogisticRegressionModel()

train_output = model.train(frame, 'Class', ['Sepal_Length', 'Petal_Length'],
                           num_classes=3, optimizer='LBFGS', compute_covariance=True)

train_output.summary_table

# The covariance matrix is the inverse of the Hessian matrix for the trained model. 
# The Hessian matrix is the second-order partial derivatives of the model’s log-likelihood function
train_output.covariance_matrix.inspect()

# Use the model to make predictions
predicted_frame = model.predict(frame, ['Sepal_Length', 'Petal_Length'])

predicted_frame.inspect()

# Test the model
test_metrics = model.test(frame, 'Class', ['Sepal_Length', 'Petal_Length'])
test_metrics

