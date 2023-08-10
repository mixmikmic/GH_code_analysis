get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
#from termcolor import colored
from numbers import Number
from scipy import stats
from pandas import plotting

LARGE_FILE = "loan_data/loan_data_complete.csv"
CHUNKSIZE = 100000 # processing 100,000 rows at a time
'''
def process_frame(df):
        # process data frame
        return len(df)

if __name__ == '__main__':
        reader = pd.read_table(LARGE_FILE, chunksize=CHUNKSIZE)

        result = 0
        for df in reader:
                # process each data frame
                result += process_frame(df)

        print("There are %d rows of data"%(result))
'''

reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE, low_memory=False)
loan_data = pd.DataFrame()
frames = []

# Select data without current Status
for df in reader:
    frames.append(df[df["loan_status"] != "Current"])
loan_data = pd.concat(frames)
    

loan_data.to_csv("loan_data_seperate_current.csv", sep=',')

data_with_out_currents = len(loan_data)

reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE, low_memory=False)
loan_data_current = pd.DataFrame()
frames_current = []

# only select data with current loan_status
for df in reader:
    frames_current.append(df[df["loan_status"] == "Current"])
loan_data_current = pd.concat(frames_current)
    

loan_data_current.to_csv("loan_data_current.csv", sep=',')

data_with_current = len(loan_data_current) 

print("Total Number of data is: " ,(data_with_current + data_with_out_currents))
print("It suppose to have 1646801 amount of data points")
print((data_with_current + data_with_out_currents) == 1646801)



