get_ipython().magic('matplotlib inline')

parameters_lists = [['request', 'template_name', 'context', 'content_type', 'status', 'using'],
['field', 'request', 'params', 'model', 'model_admin', 'field_path'],
['request', 'label', 'model', 'field_name', 'compress', 'using'],
['addr', 'port', 'wsgi_handler', 'ipv6', 'threading', 'server_cls'],
['data', 'offset', 'size', 'shape', 'as_memoryview'],
['request', 'label', 'model', 'field_name', 'using'],
['obj', 'key', 'salt', 'serializer', 'compress'],
['s', 'key', 'salt', 'serializer', 'max_age'],
['targets', 'plan', 'state', 'fake', 'fake_initial'],
['receiver', 'sender', 'weak', 'dispatch_uid', 'apps']]

unique_parameters = set([parameter for parameters_list in parameters_lists for parameter in parameters_list]) #41 parameters, 54 total
parameters_frequency = []

for parameters_list in parameters_lists:
    parameters_frequency_row = []
    for parameter in unique_parameters:
        if parameter in parameters_list: 
            parameters_frequency_row.append(1)
        else:
            parameters_frequency_row.append(0)
    parameters_frequency.append(parameters_frequency_row)

import pandas as pd

parameters = pd.DataFrame(parameters_frequency, columns=unique_parameters)
parameters.head()

parameter_support_dict = {}
for column in parameters.columns:
    parameter_support_dict[column] = sum(parameters[column] > 0)
    
pd.Series(parameter_support_dict).plot(kind="bar")



