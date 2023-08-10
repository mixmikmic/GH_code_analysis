# -- Import the dataset from the microsoftml package
from microsoftml.datasets.datasets import DataSetMtCars
mtcars = DataSetMtCars()

# -- Represent the dataset as a dataframe.
mtcars = mtcars.as_df()

# -- print top rows of data to inspect the data
mtcars.head()

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
import pandas as pd
from revoscalepy import rx_lin_mod, rx_predict

# -- using rx_lin_mod from revoscalepy package
# -- create glm model with `mtcars` dataset
cars_model = rx_lin_mod(
    formula='am ~ hp + wt',
    data=mtcars)

# -- provide some sample inputs to test the model
mydata = pd.DataFrame({
    'hp':[120],
    'wt':[2.8]
})
mydata

# -- predict locally
rx_predict(cars_model, data=mydata)

# --Define an `init` function to handle service initialization --
def init():
    import pandas as pd
    from revoscalepy import rx_predict

def manualTransmission(hp, wt):
    import pandas as pd
    from revoscalepy import rx_predict
    
    # -- make the prediction use model `cars_model` and input data --
    newData = pd.DataFrame({'hp':[hp], 'wt':[wt]})
    answer = rx_predict(cars_model, newData, type='response')
    
    # -- save some files to demonstrate the ability to return file artifacts --
    answer.to_csv('answer.csv')
    # return prediction
    return answer

service_name = 'TxService'

service = client.service(service_name)        .version('1.0')        .code_fn(manualTransmission, init)        .inputs(hp=float, wt=float)        .outputs(answer=pd.DataFrame)        .models(cars_model=cars_model)        .description('My first python model')        .artifacts(['answer.csv'])        .deploy()

print(help(service))

service.capabilities()

res = service.manualTransmission(120, 2.8)

# -- Pluck out the named output `answer` as defined during publishing and print --
print(res.output('answer'))

# -- Retrieve the URL of the swagger file for this service.
cap = service.capabilities()
swagger_URL = cap['swagger']
print(swagger_URL)

client.delete_service('TxService', version='1.0')

