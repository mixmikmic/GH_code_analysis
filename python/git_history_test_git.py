import githistoryvis as ghv

import os

path =  os.getcwd() # put here the desired git repo path

gt = ghv.git_history(path)

gt.get_history()

# new compact version
gt = ghv.git_history(path, get_history=True)

with open('gitoutput', 'r') as file:
    data = file.read()
gt.get_history(gitcommitlist=data)

gt.definedatamatrix()
gt.datamatrix

import matplotlib
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

gt.plot_history_df(plt,gt.datamatrix,size= 300, figsize = [12,10.5])
gt.plot_history_df(plt,gt.datamatrix,size= 300, figsize = [12,10.5],outpath=path+os.sep+'images/complete_visual_history.png')

# filtering the history on:
# a commit range
plot_df_commit_range = gt.datamatrix.ix[:,'a4cb9a1':'1222c5e']
gt.plot_history_df(plt,plot_df_commit_range,size= 300, figsize= [3,10])
gt.plot_history_df(plt,plot_df_commit_range,size= 300, figsize= [3,10], outpath=path+os.sep+'images/commit_range.png')

# filtering the history on:
# a file range: all files not ending with txt
plot_df_file_range = gt.datamatrix[~gt.datamatrix.index.str.contains('txt$')]
gt.plot_history_df(plt,plot_df_file_range,size= 300, figsize= [11.5,8.5])
gt.plot_history_df(plt,plot_df_file_range,size= 300, figsize= [11.5,8.5], outpath=path+os.sep+'images/file_range.png')

# filtering the history on:
# a commit range AND a file range: all files not ending with txt
plot_df_commit_file_range = gt.datamatrix.ix[:,'a4cb9a1':'1222c5e'][~gt.datamatrix.index.str.contains('txt$')]
gt.plot_history_df(plt,plot_df_commit_file_range,size= 300,figsize= [3.5,8.5])
gt.plot_history_df(plt,plot_df_commit_file_range,size= 300,figsize= [3.5,8.5],outpath=path+os.sep+'images/commit_file_range.png')

# filtering the history on:
# a commit range AND a file range: all files not ending with txt
plot_df_state_filter = gt.datamatrix[gt.datamatrix[gt.datamatrix.columns[-1]] != 'N']
gt.plot_history_df(plt,plot_df_state_filter,size= 300,figsize= [11,6])
gt.plot_history_df(plt,plot_df_state_filter,size= 300,figsize= [11,6],outpath=path+os.sep+'images/state_filter.png')

