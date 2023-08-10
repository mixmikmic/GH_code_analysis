import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data_frame = pd.read_csv("product-market-fit-data-table.csv", 
                        header=6, 
                        names=["run", 
                              "traits", 
                              "strategy", 
                              "features", 
                              "ticks", 
                              "sales", 
                              "effort"])

data_frame.head()

mean_analysis = data_frame.groupby(["strategy"]).mean().drop(["run", "traits", "features"], 1)
mean_analysis

