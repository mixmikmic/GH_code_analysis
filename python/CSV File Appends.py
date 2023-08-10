import pandas as pd
import csv
import os
from collections import deque

import settings_skyze

market = "BiTcoin"
file_path = os.path.join(settings_skyze.data_file_path, "%s.csv" % market)
print(file_path)

top = pd.read_csv(file_path,nrows=0)

headers = top.columns.values
endDate = headers[0][:11]
print("End: "+endDate)

with open(file_path, 'r') as f:
    q = deque(f, 1)
print("start: "+q[0][:11])



