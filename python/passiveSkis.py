get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

import os, sys

SUBDIR='./data/'

# get the current directory and files inside 
print(os.getcwd()); print(os.listdir( SUBDIR ));

UpdatedHalbach040202ends = pd.read_excel(SUBDIR+"Updated Halbach 4x2x2 ski at each end.xls")

skieval_edit = pd.read_excel(SUBDIR+"ski-eval.xlsx.xlsx")

skieval = pd.read_excel(SUBDIR+"ski-eval.xlsx")

skieval2d = pd.read_excel(SUBDIR+"ski-eval-2d.xlsx")

skieval_edit

skieval2d_periodic = pd.read_excel(SUBDIR+"ski-eval_2d_periodic_abridged.xlsx")

skieval2d_periodic

ax1 = skieval2d_periodic.ix[skieval2d_periodic['inch']==1].plot.area(x="m/s",y="drag",color="Red",label="1 in")

ax2 = skieval2d_periodic.ix[skieval2d_periodic['inch']==2].plot.area(x="m/s",y="drag",color="Green",label="2 in",ax=ax1)

ax3 = skieval2d_periodic.ix[skieval2d_periodic['inch']==3].plot.area(x="m/s",y="drag",color="Blue",label="3 in",ax=ax2)

ax5 = skieval2d_periodic.ix[skieval2d_periodic['inch']==5].plot.area(x="m/s",y="drag",color="Purple",label="5 in",ax=ax3)

skieval2d_periodic.ix[skieval2d_periodic['inch']==1]["lift"]



