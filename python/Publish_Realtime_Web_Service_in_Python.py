# -- Import the dataset from the microsoftml package
from microsoftml.datasets.datasets import DataSetAttitude
attitude = DataSetAttitude()

# -- Represent the dataset as a dataframe.
attitude = attitude.as_df().drop('Unnamed: 0', axis = 1).astype('double')

# -- print top rows of data to inspect the data
attitude.head()

# -- Import the DeployClient and MLServer classes from the azureml-model-management-sdk package.
from azureml.deploy import DeployClient
from azureml.deploy.server import MLServer

# -- Define the location of the ML Server --
# -- for local onebox for Machine Learning Server: http://localhost:12800
# -- Replace with connection details to your instance of ML Server. 
HOST = 'http://localhost:12800'
context = ('admin', 'YOUR_ADMIN_PASSWORD')
client = DeployClient(HOST, use=MLServer, auth=context)

# -- import the needed classes and functions
from revoscalepy import rx_lin_mod, rx_predict

# -- using rx_lin_mod from revoscalepy package
# -- create glm model with `attitude` dataset
df = attitude
form = "rating ~ complaints + privileges + learning + raises + critical + advance"
model = rx_lin_mod(form, df, method = 'regression')

# -- provide some sample inputs to test the model
myData = df.head(1)

# -- predict locally
print(rx_predict(model, myData))

# Import the needed classes and functions
from revoscalepy import rx_serialize_model

# Serialize the model with rx_serialize_model
s_model = rx_serialize_model(model, realtime_scoring_only=True)

service = client.realtime_service("LinModService")         .version('1.0')         .serialized_model(s_model)         .description("This is a realtime model.")         .deploy()

print(help(service))

service.capabilities()

# -- Consume the service. Pluck out the named output, outputData. --

print(service.consume(df.head(1)).outputs['outputData'])

# -- Retrieve the URL of the swagger file for this service.
cap = service.capabilities()
swagger_URL = cap['swagger']
print(swagger_URL)

# Uncomment this line if you want to delete the service now.
#client.delete_service('LinModService', version='1.0')

