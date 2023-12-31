# -- Import the DeployClient and MLServer classes from the azureml-model-management-sdk package.
from azureml.deploy import DeployClient
from azureml.deploy.server import MLServer

# -- Define the location of Machine Learning Server --
# -- for local onebox: http://localhost:12800
HOST = 'http://localhost:12800'
context = ('admin', 'YOUR_ADMIN_PASSWORD')
client = DeployClient(HOST, use=MLServer, auth=context)

# Delete any existing service by this name.
client.delete_service('TxService', version='1.0')

# Read in the mtcars dataset you'll use when modeling
from microsoftml.datasets.datasets import DataSetMtCars
mtcars = DataSetMtCars()

# -- Represent the dataset as a dataframe.
mtcars = mtcars.as_df()

# Create and run a generalized linear model locally 
import pandas as pd
from revoscalepy import rx_lin_mod, rx_predict

cars_model = rx_lin_mod(
    formula='am ~ hp + wt',
    data=mtcars)

# Define an `init` function to handle service initialization
def init():
    import pandas as pd
    from revoscalepy import rx_predict

# Produce a prediction function called manualTransmission 
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

# Publish the linear model as a Python web service with 
service_name = 'TxService'

service = client.service(service_name)        .version('1.0')        .code_fn(manualTransmission, init)        .inputs(hp=float, wt=float)        .outputs(answer=pd.DataFrame)        .models(cars_model=cars_model)        .description('My first python model')        .artifacts(['answer.csv'])        .deploy()

# -- List all services by this name --
client.list_services('TxService')

# -- Return the web service object for TxService 1.0 and assign to svc.
svc = client.get_service('TxService', version='1.0')
print(svc)

# -- Learn more about the service.
print(help(svc))

# -- View the service object's capabilities/schema.
svc.capabilities()
# -- Notice the available public functions.

# -- Import the dataset from the microsoftml package
from microsoftml.datasets.datasets import get_dataset
mtcars = get_dataset('mtcars')

# -- Represent the dataset as a dataframe.
mtcars = mtcars.as_df()

# -- Define the data for the execution.
records = mtcars[['hp', 'wt']]

batch = svc.batch(records)

batch = batch.start()

# Get the batch execution id.
id = batch.execution_id

print("The execution_id of this batch service is {}".format(id))

# Check the results every second until the task finishes or fails. 
# Assign returned results to the 'batchres' batch result object.
import time

batchRes = None
while(True):
    batchRes = batch.results()
    print(batchRes)
    if batchRes.state == "Failed":
        print("Batch execution failed")  
        break
    if batchRes.state == "Complete": 
        print("Batch execution succeeded")  
        break
    print("Polling for asynchronous batch to complete...")
    time.sleep(1)

for i in range(batchRes.completed_item_count):
    print("The result for {} th row in the record data is: {}".          format(i, batchRes.execution(i).outputs['answer']))

# List every artifact generated by this execution index for a specific row.
# Here, each row should have a "answer.csv" file.
lst_artifact = batch.list_artifacts(1)
print(lst_artifact)

# Then, get the contents of each artifact returned in the previous list.
# The result is a byte string of the corresponding object.
for obj in lst_artifact:
    content = batch.artifact(1, obj)

# Then, download the artifacts from execution index to the current working directory  
# unless a dest = "<path>" is specified.
# Here, this file is in the working directory.
batch.download(1, "answer.csv")



