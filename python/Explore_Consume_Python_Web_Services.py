# -- Import the DeployClient and MLServer classes from the azureml-model-management-sdk package.
from azureml.deploy import DeployClient
from azureml.deploy.server import MLServer

# -- Define the location of Machine Learning Server --
# -- for local onebox: http://localhost:12800
HOST = 'http://localhost:12800'
context = ('admin', 'YOUR_ADMIN_PASSWORD')
client = DeployClient(HOST, use=MLServer, auth=context)

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

# Start interacting with the service. Let's call the function `manualTransmission`.
res = svc.manualTransmission(120, 2.8)

# -- Print Response object to inspect what was returned --
print(res)

# -- Pluck out the named output `answer` and print --
print(res.output('answer'))

# -- Retrieve the URL of the swagger file that defines this service.
cap = svc.capabilities()
swagger_URL = cap['swagger']
print(swagger_URL)

# -- Print the contents of the swagger doc
print(svc.swagger())









