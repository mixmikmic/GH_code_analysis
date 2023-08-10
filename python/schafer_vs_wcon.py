get_ipython().run_line_magic('matplotlib', 'inline')

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import sciunit
from owtests import OW_HOME
import wcon
import open_worm_analysis_toolbox as owat

OWAT_TESTS = os.path.join(OW_HOME,'tests','owtests','open-worm-analysis-toolbox')
mpl.rcParams.update({'font.size':16})

def get_schafer_features(file):
    # Load a "basic" worm from a file
    path = os.path.join(OW_HOME,'open-worm-analysis-toolbox','example_data',file)
    bw = owat.BasicWorm.from_schafer_file_factory(path)
    # Normalize the basic worm
    nw = owat.NormalizedWorm.from_BasicWorm_factory(bw)
    # Compute the features
    wf_experiment_schafer = owat.WormFeatures(nw)
    return wf_experiment_schafer

get_ipython().run_line_magic('time', "wf_experiment_schafer_1 = get_schafer_features('example_contour_and_skeleton_info.mat')")

# Doesn't work due to file structure issues
#%time wf_experiment_schafer_2 = get_schafer_features('example_video_norm_worm.mat')

# A function to convert the WCON skeleton data into the format required for owat.BasicWorm 
def basic_worm_from_wcon(wcon_data):
    bw = owat.BasicWorm()
    data = wcon_data.data
    worm_name = data.columns.levels[0][0]
    skeleton_data = [data[worm_name].iloc[frame].unstack().iloc[1:].values for frame in range(data.shape[0])]
    bw._h_skeleton = skeleton_data
    return bw

def get_wcon_features(file):
    path = os.path.join(OWAT_TESTS,file)
    data = wcon.WCONWorms.load_from_file(path)
    bw = basic_worm_from_wcon(data)
    nw = owat.NormalizedWorm.from_BasicWorm_factory(bw)
    wf = owat.WormFeatures(nw)
    return wf

# Doesn't work due to formatting issues
#file = 'experiment/nca-1 (gk9) nca-2 (gk5) nRHO-1 QT309 on food L_2011_11_09__12_02___3___6.wcon'
#%time wf_experiment_wcon = get_wcon_features(file)

file = 'model/worm_motion_log.wcon'
get_ipython().run_line_magic('time', 'wf_model_wcon = get_wcon_features(file)')

features = list(wf_experiment_schafer_1._features.keys())
df = pd.DataFrame(index=features,columns=['wcon_model','wcon_experiment','schafer_experiment'])

def fill_df(wf,df,column):
    n_valid = 0
    for feature,value in wf._features.items():
        try:
            result = pd.Series(value.value).mean()
            if not np.isnan(result):
                df.loc[feature,column] = result
                #print(key,result)
                n_valid += 1
        except:
            pass
    print("%d valid features found for %s" % (n_valid,column))
    return df.astype('float')
    
df = fill_df(wf_model_wcon,df,'wcon_model')
#df = fill_df(wf_experiment_wcon,df,'wcon_experiment')
df = fill_df(wf_experiment_schafer_1,df,'schafer_experiment')
df.head()

feature_name = 'posture.amplitude_ratio'
model_mean = pd.Series(wf_model_wcon.get_features(feature_name).value).mean()
experiment_mean = pd.Series(wf_experiment_schafer_1.get_features(feature_name).value).mean()
print(model_mean,experiment_mean)

df.plot.scatter(x='schafer_experiment',y='wcon_model')

df.plot.scatter(x='schafer_experiment',y='wcon_model')
plt.xlim(-10,10)
plt.ylim(-10,10)

