get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (16, 6)
import numpy as np
import matplotlib.pyplot as plt
import json
from types import SimpleNamespace as Namespace

file_directory = '../../data/Pin hole tip.json'

# Pin hole tip.json 
# Scallop tip.json


json_data=open(file_directory).read()

# print(json_data[:100])

x = json.loads(json_data, object_hook=lambda d: Namespace(**d))
print(len(x.data))
# print(x.data[100].event.content)

xRow = list()
start = 6000
limit = 8000 #len(x.data)
variable = 'quaternion' #quaternion acceleration
for index in range(start,limit):
#     print(x.data[index].event.variable == 'acceleration')
    if(x.data[index].event.variable == 'acceleration'):#quaternion
        xRow.append(x.data[index].event.content[0])
        
mean = sum(xRow)/len(xRow)
print('mean:', mean)
    
    
plt.plot(range(0,limit if len(xRow)>limit else len(xRow)), xRow[:limit], '-')













