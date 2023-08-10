BIGML_USERNAME = '' # fill in your username between the quotes
BIGML_API_KEY = '' # fill in your API key
BIGML_AUTH = 'username=' + BIGML_USERNAME + ';api_key=' + BIGML_API_KEY # leave as it is
print "Authentication variables set!"

from bigml.api import BigML

# Assuming you installed the BigML Python wrappers (with the 'pip install bigml' command, see above)
# Assuming BIGML_USERNAME and BIGML_API_KEY were defined as shell environment variables
# otherwise: api=BigML('your username here','your API key here',dev_mode=True)

api=BigML(BIGML_USERNAME, BIGML_API_KEY, dev_mode=True) # use BigML in development mode for unlimited usage
print "Wrapper ready to use!"

source = api.create_source('s3://bigml-public/csv/iris.csv', {"name": "Iris source"})
print "'source' object created!"

api.ok(source) # shows "True" when source has been created

print "https://bigml.com/dashboard/"+str(source['resource'])+"?"+BIGML_AUTH

dataset = api.create_dataset(source, {"name": "Iris dataset"})
api.ok(dataset)
print "Dataset ready and available at https://bigml.com/dashboard/"+str(dataset['resource'])+"?"+BIGML_AUTH

model = api.create_model(dataset)
print "'model' object created!"

api.ok(model) # making sure the model is ready
print "https://bigml.com/dashboard/"+str(model['resource'])+"?"+BIGML_AUTH

# the strings below correspond to the headers of the iris.csv file we used to create the model
new_input = {"sepal length": 4.8, "sepal width": 4.5, "petal length": 1.0, "petal width": 0.7}
print "'new_input' object created!"

prediction = api.create_prediction(model, new_input)
print "Prediction: ",prediction['object']['output']
print "Confidence: ",prediction['object']['confidence']

