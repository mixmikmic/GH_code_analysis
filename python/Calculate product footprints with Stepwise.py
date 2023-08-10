cd /Users/marie/Desktop

## Import terminated results
import csv
import numpy as np
with open("BLCI.csv",'r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
BLCI = np.array(list(data)).astype('float')

## Import stepwise mid-points (mid-point methods)
import pandas as pd
data_xls = pd.read_excel('stepwise_1-6_exiobase.xlsx', 'stepwise-exiobase', index_col=None, header = None, encoding='utf-8')
##outfile ="/Users/marie/Desktop/stepwise_mid_pd.csv"
stepwise_mid = data_xls.iloc[1:102, 5:20]
##stepwise_mid.to_csv(outfile, header = None)

## Import stepwise end-points (damage vector in EUROS)
import pandas as pd
data_xls = pd.read_excel('stepwise_1-6_exiobase.xlsx', 'stepwise-exiobase', index_col=None, header = None, encoding='utf-8')
##outfile ="/Users/marie/Desktop/stepwise_EUR_pd.csv"
stepwise_EUR = data_xls.iloc[102:117, 20:21]
##stepwise_EUR.to_csv(outfile, header = None)

## Multiply terminated results with Stepwise to Get mid-point product footprints
midpoint_footprints = np.dot(np.transpose(stepwise_mid),BLCI)

from io import StringIO
import numpy as np
s=StringIO()
np.savetxt('midpoint_footprints.csv', midpoint_footprints, fmt='%.10f', delimiter=',', newline="\n")

## Multiply mid-point product footprints with damage vector to get end-point product footprints
endpoint_footprints = np.dot(np.transpose(stepwise_EUR),midpoint_footprints)

from io import StringIO
import numpy as np
s=StringIO()
np.savetxt('endpoint_footprints.csv', endpoint_footprints, fmt='%.10f', delimiter=',', newline="\n")

## DO NOT EXECUTE
## Make a terminated matrix with shape (8313,8313) , to be used later to integrate stepwise in extended exiobase
import csv
import numpy as np
a = np.zeros(shape=(8212,8313))
b = np.zeros(shape=(101,101))
frame1 = np.concatenate((BLCI, b), axis=1)
terminated_matrix = np.concatenate((a, frame1), axis=0)

from io import StringIO
import numpy as np
s=StringIO()
np.savetxt('terminated_matrix.csv', terminated_matrix, fmt='%.10f', delimiter=',', newline="\n")



