import pandas as pd
np=pd.np
from sdd_api.api import Api
from credentials import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
api = Api(username=username, password=password, client_id=client_id, client_secret=client_secret)

dfs=api.get_dataframe('dfs_salaries')
dfs.sample(4)

dfs.describe()

dfs.groupby(["position"]).agg([max])

dfs.groupby("position")[['position','dk_points','fd_points','yh_points']].agg(max).plot(kind="bar")

dfs2014_present=api.get_dataframe('dfs_salaries',season_start=2014)
dfs2014_present.sample(10)

