get_ipython().run_line_magic('matplotlib', 'inline')

# Using Web Map Tile Service (WMPS) to get satellite images
# Following http://www.net-analysis.com/blog/cartopyimages.html
import cartopy.feature as cfeature
import cartopy.crs as ccrs
#from cartopy.io.img_tiles import OSM
#import cartopy.feature as cfeature
#from cartopy.io import shapereader
#from cartopy.io.img_tiles import StamenTerrain
#from cartopy.io.img_tiles import GoogleTilesimport
import cv2
import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
from owslib.wmts import WebMapTileService
#from PIL import Image
#from pykrige.compat import GridSearchCV
from pykrige.rk import RegressionKriging
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ARDRegression, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
# Adjust size of figures
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# ALL parameters are defined in this cell
# Define data and "pre" preprocessing parameters
# Run next cell for list of features
# latlon_ll = latitude, longitude of map lower-left corner in degrees
# latlon_ur = latitude, longitude of map upper-right corner in degrees
# If latlon_ll and/or latlon_ur are empty, then they are set to the minimum 
# and maximum lat, lon across all data points
# If both latlon_ll and latlon_ur are provided and filter_latlons = True, then
# only data points within latlon_ll and latlon_ur are included in the feature
# array, X, and the target variable, y.
latlon_ids = [6, 7]
filter_latlons = True
latlon_ll = [38.8, -124.6]
latlon_ur = [42.2, -119.8]
# with buffer
#latlon_ll = [38.9, -124.5]
#latlon_ur = [42.1, -119.9]
mapxy_projection = "mercator"
remove_outliers = True
rm_feature_inds = [5]
rm_feature_upth = [100]
rm_feature_loth = [None]

# Preprocessing
prepro_method = "standardscaler"
prepro_params = {"copy": False}
prepro_features = False
prepro_target = False

# Define Regression (reg) and Regression Kriging (rk) parameters
# reg_feature_ids = the indices of the features to include in regression, e.g. to
# choose whether to include the lat/lon values in the regression or not, etc.
# rk_n_closest_points = # of closest points to include in Kriging estimate
reg_feature_ids = [0, 1, 2, 3, 4, 5, 6, 7]
kri_n_closest_points = 8
kri_method = "ordinary"  # ordinary or universal
kri_varmod = "spherical"  # linear, power, gaussian, spherical, exponential, hole-effect
kri_nlags = 40
kri_weight = True

# Define Plotting parameters
# Map params
# latlon_buf is used to add space around edges of map by inputing
# a single latitude and longitude value, in degrees
# Note these two values only change the area shown in the map, 
# they do NOT effect the filtering of data points, for example
latlon_buf = [0.1, 0.1]
# Satellite image params
# Select WTMS Tile Service
# URL = NASA GIBS URL for wtms files
# layers = MODIS true color and snow RGB keys
# data_str = Approximate date of returned images (e.g. URL "best")
URL = 'http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi'
layers = ['MODIS_Terra_SurfaceReflectance_Bands143',         'MODIS_Terra_CorrectedReflectance_Bands367']
date_str = '2010-07-12'
# Figure params
fig_dpi = 250
figsize_y = 8
plot_datapoints = True
plot_dptype = "scatter"
plot_dpcolormap = "magma"
plot_dpcolorbar = False
savefig_on = True
savefig_name = "NCal_modis_olric.png"

# Define Preprocessing parameters
# Image feature extraction params, NO data points, colorbar, etc.!
# i.e. only use images constructed with plot_datapoints = plot_dpcolorbar = False 
# If feature_norm = True, normalize features: subtract mean then divide by L2 norm
imagefe_name = "NCal_modis_olric.png"

# Function definitions
# Collect input data
def get_inputs_dict(data_dict="California Housing", keys_to_get=[], mapxy_projection="mercator"):
    """Get input data stored in dict, inputs, which contains:
    1) Feature variables [# Samples, # Features], key='X'
    2) Feature names [# Features], key='feature_names'
    3) Target variable [# Samples], key='y'
    4) *Geographic location variables [# of Samples, # Coordinates (e.g. x,y)], key='xy'
    5) Dataset description [single string of text], key='DESCR'
    * Note, the cartopy package MUST be installed to build geographic (xy) variables.  If this requirement 
    is met, two additional variables will be appended to inputs: a) a "geodetic" coordinate reference frame 
    (CRF) object, key="geodetic_CRS", and b) a "cartesian" map projection CRF, key="mapxy_CRS". Currently, 
    only mapxy_projection="mercator" is supported.
    Input data_dict is a string "pointing" to a method for loading a data dictionary.
    Currently only California Housing dataset is available: data_dict="California Housing"
    """
    inputs = {}
    if data_dict == "California Housing":
        if len(keys_to_get) == 0:
            keys_to_get = ['X', 'y', 'xy', 'feature_names', 'DESCR']
        # Load data and extract selected features
        # (Down)load California housing data
        chd = fetch_california_housing()
        # Extract features and target from input dataset
        if "X" in keys_to_get:
            inputs['X'] = chd.data
        if "y" in keys_to_get:
            inputs['y'] = chd.target
            #inputs['y'] = inputs['y'].reshape(inputs['y'].shape[0], -1)
        if "xy" in keys_to_get:
            # Extract separate feature array that contains the lat/lon features 
            # converted to x/y coordinates.
            inputs['geodetic_CRS'] = ccrs.Geodetic()
            if mapxy_projection == "mercator":
                inputs['mapxy_CRS'] = ccrs.Mercator()
            inputs['xy'] = convert_latlon_to_xy(chd.data[:, latlon_ids], inputs['mapxy_CRS'],                     inputs['geodetic_CRS']
            )
        if "feature_names" in keys_to_get:
            inputs['feature_names'] = chd.feature_names
        if "DESCR" in keys_to_get:
            inputs['DESCR'] = chd.DESCR
    else:
        print("Error, unrecognized data_dict value = {}".format(data_dict))
    return inputs

# Preprocessing functions
# Get analysis dict
def get_analysis_dict(inputs, keys_to_copy=['X', 'y', 'xy']):
    """Get the analysis dictionary, analysis, which is instantiated here with a copy of the X, y, and
    xy variables from the input dict, inputs, if they exist.  This dict is used as a container for all
    preprocessing/modeling data. Note that all subsequent data stored in the analysis dict should be 
    stored under either the preprocessing or modeling key.
    """
    return {k: v.copy() for k, v in inputs.items() if k in keys_to_copy}

# Filter data rows by latitude and longitude
def prepro_filter_by_latlon(data, latlon_ids, latlon_ll, latlon_ur):
    """Return subset (i.e. filter) of rows of feature array, X, target, y, (and optionally xy) based 
    on lower-left and upper-right latitude and longitude pairs, latlon_ll and latlon_ur, respectively.
    Filter is applied if filter_latlons=True and latlon_ll and latlon_ur are each of length 2.  
    latlon_ids = a two-element list containing the column indices corresponding to latitude and
    longitude variables in the feature array, X.  Returns input dict, data, with filtered X, y, and xy.
    """ 
    latlon_mask = (data['X'][:, latlon_ids[0]] >= latlon_ll[0]) &             (data['X'][:, latlon_ids[0]] <= latlon_ur[0]) &             (data['X'][:, latlon_ids[1]] >= latlon_ll[1]) &             (data['X'][:, latlon_ids[1]] <= latlon_ur[1])
    data['X'] = data['X'][latlon_mask, :]
    data['y'] = data['y'][latlon_mask]
    if "xy" in data:
        data['xy'] = data['xy'][latlon_mask, :]
    return data

def prepro_remove_outliers(data, rm_feature_inds, rm_feature_loth, rm_feature_upth):
    for ind, fid in enumerate(rm_feature_inds):
        if rm_feature_upth[ind]:
            out_mask = data['X'][:, fid]<rm_feature_upth[ind]
            data['X'] = data['X'][out_mask, :]
            data['y'] = data['y'][out_mask]
            data['xy'] = data['xy'][out_mask, :]
        if rm_feature_loth[ind]:
            out_mask = data['X'][:, fid]>rm_feature_loth[ind]
            data['X'] = data['X'][out_mask, :]
            data['y'] = data['y'][out_mask]
            data['xy'] = data['xy'][out_mask, :]
    return data

# Model Score plotting functions
def plot_scores_image(fig, ax, array, aspect=1):
    cax = ax.imshow(array, aspect=aspect, origin="lower")
    fig.colorbar(cax)

def plot_scores_hist_cascade(ax, array, bin_edges):
    hist_height_adj = 0.95
    bin_width_adj = 1
    nr, nc = array.shape
    arr_means = np.mean(array, axis=0)
    arr_stds = np.std(array, axis=0)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for ind in range(nc):
        hist, _ = np.histogram(array[:, ind], bins=bin_edges)
        ax.bar(bin_centers, hist_height_adj*hist/hist.max(), align='center',                width=bin_width_adj*bin_width, bottom=ind
        )
        ax.plot([arr_means[ind], arr_means[ind]], [ind, ind+1], 'k', linewidth=2)
        ax.plot([arr_means[ind]-arr_stds[ind], arr_means[ind]+arr_stds[ind]],                 [ind, ind], 'k', linewidth=2
        )
        ax.plot([bin_edges[0], bin_edges[-1]], [ind, ind], 'k', linewidth=1)
    ax.set_xlim([bin_edges[0], bin_edges[-1]])
    ax.set_ylim([0, nc])
    
# Map-related functions
# Convert lat/lon features from geographic coordinates to x/y coordinates
def convert_latlon_to_xy(latlon, crs1, crs2):
    """Convert latitude/longitude array in x/y array using coordinate reference
    systems, crs1 and crs2.  Returns array of x/y values
    """
    xy = np.zeros((latlon.shape[0], 2))
    for ri, (lat, lon) in enumerate(latlon):
        xy[ri,:] = crs1.transform_point(lon, lat, crs2)
    return xy

# Load inputs dictionary containing input variables from source
inputs = get_inputs_dict()
print("Feature names (in order): \n{}\n".format(inputs['feature_names']))
#print("Data description: \n{}\n".format(inputs['DESCR']))

# Load analysis dictionary containing preprocessing and modeling variables, including
# a copy of the feature array, X, the target variable, y, and the spatial variable, xy.
# First, filter the data by selecting a subset of the rows of X, y, and (optionally) xy, 
# based on user provided lower-left and upper-right lat/lons pairs. Filter is applied if 
# filter_latlons=True and latlon_ll and latlon_ur are each of length 2
analysis = get_analysis_dict(inputs)
analysis = prepro_filter_by_latlon(analysis, latlon_ids, latlon_ll, latlon_ur)
if remove_outliers:
    analysis = prepro_remove_outliers(analysis, rm_feature_inds, rm_feature_loth, rm_feature_upth)

print(inputs['X'].shape, analysis['X'].shape)



# Plot target variable (median housing price) vs specified feature variable
plt_featid = 5
feature_name = inputs['feature_names'][plt_featid]
#plt.scatter(analysis['X'][:, plt_featid], analysis['y'], color='black')
#plt.hist(analysis['X'][:, plt_featid]) #, bins=np.arange(0, 40000, 1000))
plt.hist(analysis['X'][:, plt_featid])
#plt.ylim([0, 10])
#plt.title("Median Housing Price vs. {}".format(feature_name))\
plt.title("Histogram Count of {}".format(feature_name))
plt.show()

# Plot target residuals (difference between model and data) vs specified 
# feature variable
#chd_y_res = chd_y_test - chd_y_pred
#resmax_ind = chd_y_res.argmax()
#plt.scatter(inputs['X']_train[:,plt_featid], chd_y_train,  color='black')
#plt.scatter(inputs['X']_test[:,plt_featid], chd_y_test,  color='red')
#plt.scatter(inputs['X']_test[:,plt_featid], chd_y_res, color='blue')
#plt.title("Residual Price vs. {}".format(feature_name))
#plt.show()





# Select hyperparameters using cross-validation
# Modified after http://scikit-learn.org/stable/auto_examples/

# Parameters
# List of regress(ion) methods: LinearRegression, Ridge, Lasso, ElasticNet,
# SGDRegressor, ARDRegression, LinearSVR, SVR, BayesianRidge, RANSACRegressor, 
# GaussianProcessRegressor, AdaBoostRegressor, GradientBoostingRegressor, 
# Number of random trials
NUM_TRIALS = 50
model_method = "Validate-Test"
val_nsplits = 3
test_nsplits = 4
scale_grid = [None, preprocessing.StandardScaler()]
regress_grid = [LinearRegression(), ElasticNet(), LinearSVR()]
#C_grid = [0.1, 1]
#epsilon_grid = [0.05, 0.1]
alpha_grid = [1e-4, 1e-3]
l1rat_grid = [0.01, 0.1, 0.5, 0.9, 0.99]
p_grid = [         #{
        #'scale': [scale_grid[1]], \
        #'regress': [regress_grid[0]]
        #}, \
        {
        'scale': [scale_grid[1]], \
        'regress': [regress_grid[1]], \
        'regress__alpha': alpha_grid, \
        'regress__l1_ratio': l1rat_grid
        }
        #{
        #'scale': [scale_grid[1]], \
        #'regress': [regress_grid[2]], \
        #'regress__C': C_grid, \
        #'regress__epsilon': epsilon_grid
        #}
]
# Write figures to file as string savefig*, use empty string to show figure
# instead of saving it
savefig_val = "val_trainval_scores_elanet.png"
savefig_test = "test_valtest_scores_elanet.png"

# Make pipeline
pipe = Pipeline([('scale', preprocessing.StandardScaler()),                 ('regress', LinearRegression())]
)
# Perform model selection and/or evaluation for a set of random trials.
# There are two modes of operation: 1) "Validate": Model exploration/selection using grid 
# search to evaluate different models/hyper-parameters via un-nested Cross-Validation (UCV). 
# This option only splits the data into training and validation sets, and provides score/error 
# estimates across a grid of different methods/hyper-parameters, but these estimates may be 
# significantly biased, so this option is best for parameter grid tuning, early-stage model
# exploration, etc., but score estimates etc. are not good for evaluating model performance
# 2) "Test": Model selection AND performance evaluation using grid search and CV for 
# validation in an "inner loop", and then using (nested) CV in an "outer loop", to evalute 
# the inner, validation model's expected performance on new, "unseen" samples.  This option
# splits the data into training, validation, and testing sets and picks an "optimal" model 
# internally (i.e. in the inner loop), based only on the training and validation datasets.
# The model in this case encompasses the entire GridSearch object, including the chosen
# CV method and the parameter grid.
# 3) "Validate-Test": Runs both 1 and 2 above for model exploration, debugging, etc.

# If model_method starts with "Validate", store v_train_scores and v_val_scores for each
# loop in lists 
# If model_method ends with "Test", store t_val_scores and t_test_scores in lists

if model_method.startswith("Validate"):
    v_train_scores = []
    v_val_scores = []
    v_val_score_bests = []
    v_val_score_ranks = []
if model_method.endswith("Test"):
    t_val_scores = []
    t_test_scores = []

print("Running Model {} method".format(model_method))
for i in range(NUM_TRIALS):
    print("Trial #{}".format(i))
    # First part of Validation Step no matter the method: instantiate 1) CV class that
    # defines validation "loop" sampling strategy, and 2) Grid search CV class for 
    # estimation of optimal modeleling step(s)/hyper-parameter(s)
    val_cv = KFold(n_splits=val_nsplits, shuffle=True, random_state=i)
    clf = GridSearchCV(estimator=pipe, param_grid=p_grid, cv=val_cv)
    if model_method.startswith("Validate"):
        print("Validating...")
        clf.fit(analysis['X'], analysis['y'])
        if i == 0:
            v_params = clf.cv_results_['params']
        key_suf = ["_train_score", "_test_score"]
        for ns in range(val_nsplits):
            v_train_scores.append(clf.cv_results_["split"+str(ns)+key_suf[0]])
            v_val_scores.append(clf.cv_results_["split"+str(ns)+key_suf[1]])
        v_val_score_bests.append(clf.best_score_)
        v_val_score_ranks.append(clf.cv_results_['rank_test_score'])
            
    if model_method.endswith("Test"):
        # Testing: Evaluate model performance using outer loop sampling strategy defined
        # by test_cv and inner loop as defined for "Validate" above
        print("Testing...")
        test_cv = KFold(n_splits=test_nsplits, shuffle=True, random_state=i)
        t_scores = cross_validate(clf, X=analysis['X'], y=analysis['y'], cv=test_cv)
        t_val_scores.append(t_scores["train_score"])
        t_test_scores.append(t_scores["test_score"])

if model_method.startswith("Validate"):
    v_train_scores = np.array(v_train_scores)
    v_val_scores = np.array(v_val_scores)
    v_val_score_bests = np.array(v_val_score_bests)
    v_val_score_ranks = np.array(v_val_score_ranks, dtype="int")
    vt_score_means = v_train_scores.mean(axis=0)
    vv_score_means = v_val_scores.mean(axis=0)
    vv_score_rank_means = v_val_score_ranks.mean(axis=0)
    vt_score_stds = v_train_scores.std(axis=0)
    vv_score_stds = v_val_scores.std(axis=0)
    vv_score_rank_stds = v_val_score_ranks.std(axis=0)
if model_method.endswith("Test"):
    t_val_scores = np.array(t_val_scores)
    t_test_scores = np.array(t_test_scores)
    tv_score_means = t_val_scores.mean(axis=0)
    tt_score_means = t_test_scores.mean(axis=0)
    tv_score_stds = t_val_scores.std(axis=0)
    tt_score_stds = t_test_scores.std(axis=0)
print("Finished!")

print("Total # of Samples Included in Regression = {}".format(analysis['X'].shape[0]))
if model_method.startswith("Validate"):
    print("GridCV Parameter grid: \n{}".format(v_params))
    print(v_train_scores.shape)
    print(vv_score_rank_means, vv_score_rank_stds)
    # Make image plot and/or histogram plot from un-nested CV Gridsearch for
    # 1) training scores, v_train_scores, and 2) validation scores, v_val_scores
    # Set aspect ratio of image plot
    v_aspect = 10
    # Calculate bin locations
    bin_step = 0.01
    bin_min = np.min([v_train_scores.min(), v_val_scores.min()])
    bin_max = np.max([v_train_scores.max(), v_val_scores.max()])
    bin_edges = np.arange(bin_min, bin_max+bin_step, bin_step)
    # Make figure showing score values versus fold/trial, estimator, both as 
    # "raw" images and as histograms
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    plot_scores_image(fig, ax1, v_train_scores.T, aspect=v_aspect)
    ax2 = plt.subplot(2, 2, 2)
    plot_scores_hist_cascade(ax2, v_train_scores, bin_edges)
    ax3 = plt.subplot(2, 2, 3)
    plot_scores_image(fig, ax3, v_val_scores.T, aspect=v_aspect)  
    ax4 = plt.subplot(2, 2, 4)
    plot_scores_hist_cascade(ax4, v_val_scores, bin_edges)
    if len(savefig_val) > 0:
        fig.savefig(savefig_val)
    else:
        plt.show()
    # Make plot of estimator rank versus fold/trial and estimator, as an image
    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    plot_scores_image(fig, ax1, v_val_score_ranks.T, aspect=np.floor(v_aspect/2))
    print(v_val_score_ranks.shape)
    
if model_method.endswith("Test"):
    # Make image plot and/or histogram plot from nested CV Gridsearch for
    # 1) validation scores, t_val_scores, and 2) test scores, t_test_scores
    # Set aspect ratio of image plot
    t_aspect = 12
    # Calculate bin locations
    bin_step = 0.01
    bin_min = np.min([t_val_scores.min(), t_test_scores.min()])
    bin_max = np.max([t_val_scores.max(), t_test_scores.max()])
    bin_edges = np.arange(bin_min, bin_max+bin_step, bin_step)
    # Make figure
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    plot_scores_image(fig, ax1, t_val_scores.T, aspect=t_aspect)
    ax2 = plt.subplot(2, 2, 2)
    plot_scores_hist_cascade(ax2, t_val_scores, bin_edges)
    ax3 = plt.subplot(2, 2, 3)
    plot_scores_image(fig, ax3, t_test_scores.T, aspect=t_aspect)  
    ax4 = plt.subplot(2, 2, 4)
    plot_scores_hist_cascade(ax4, t_test_scores, bin_edges)
    if len(savefig_test) > 0:
        fig.savefig(savefig_test)
    else:
        plt.show()
    
if model_method == "Validate-Test":
    print("\nUn-Nested Validation, Training Score: Mean of Estimator Means, Estimator Means +/- St. Devs.:")
    print("{}, {} +/- {}".format(vt_score_means.mean(), vt_score_means, vt_score_stds))
    print("Un-Nested Validation, Validation Score: Mean of Estimator Means, Estimator Means +/- St. Devs.:")
    print("{}, {} +/- {}".format(vv_score_means.mean(), vv_score_means, vv_score_stds))
    print("Un-Nested Validation, Validation Best Scores: Max Best Score, Mean Best Score +/- St. Dev.:")
    print("{}, {} +/- {}".format(v_val_score_bests.max(), v_val_score_bests.mean(), v_val_score_bests.std()))
    print("\nNested Test, Validation Score: Mean of Fold Means, Fold Means +/- St. Devs.:")
    print("{}, {} +/- {}".format(tv_score_means.mean(), tv_score_means, tv_score_stds))
    print("Nested Test, Test Score: Mean of Fold Means, Fold Means +/- St. Devs.:")
    print("{}, {} +/- {}".format(tt_score_means.mean(), tt_score_means, tt_score_stds))









# Perform Regression Kriging to fit housing features with residuals adjusted to 
# match the spatial correlation structure of the target variable
# Modified after following link:
# https://github.com/bsmurphy/PyKrige/blob/master/examples/regression_kriging2d.py#L27
# The model is trained using three data variables: 1) the training features, X_train, identified 
# in reg_feature_ids, 2) the X and Y geographic coordinates obtained by transforming the
# lat/lon features using the geodetic transform, analysis['xy'], and 3) the target variable, 
# analysis['y']
# Perform regression kriging
# Rescale feature variables
X_krig = analysis['X'].copy()
X_scaler = preprocessing.StandardScaler()
X_scaler.fit(X_krig)
X_krig = X_scaler.transform(X_krig)
# Rescale spatial variables
xy_krig = analysis['xy'].copy()
xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(xy_krig)
xy_krig = xy_scaler.transform(xy_krig)
# Rescale target variable
y_krig = analysis['y'].copy()
y_krig = y_krig.reshape(-1, 1)
y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y_krig)
y_krig = y_scaler.transform(y_krig)
#print(y_krig)

# Option to convert lat/lon features to x/y
reg_latlon2xy = False
if reg_latlon2xy:
    analysis['X'][:, latlonind] = analysis['xy']

regression_model = ElasticNet(alpha=1e-3, l1_ratio=0.01)
#LinearRegression(normalize=False, copy_X=True)
m_rk = RegressionKriging(regression_model=regression_model,         method=kri_method, variogram_model=kri_varmod,         n_closest_points=kri_n_closest_points, nlags=kri_nlags, weight=kri_weight
)
# Repeated KFold Cross-Validation (CV)
rk_nsplits = 2
rk_nrepeats = 30
rk_random_state = 0
rkf = RepeatedKFold(n_splits=rk_nsplits, n_repeats=rk_nrepeats, random_state=rk_random_state)

rk_train_score = []
r_train_score = []
k_train_score = []
rk_test_score = []
r_test_score = []
k_test_score = [] 
for ind, (train_index, test_index) in enumerate(rkf.split(analysis['X'])):
    print("Working on Split #{}...".format(ind))
    X_train, X_test = X_krig[train_index], X_krig[test_index]
    xy_train, xy_test = xy_krig[train_index], xy_krig[test_index]
    y_train, y_test = y_krig[train_index], y_krig[test_index]
    m_rk.fit(X_train[:, reg_feature_ids], xy_train, y_train[:, 0])
    rk_train_score.append(m_rk.score(         X_train[:, reg_feature_ids], xy_train, y_train
    ))
    r_train_score.append(m_rk.regression_model.score(         X_train[:, reg_feature_ids], y_train
    ))
    k_train_score.append(m_rk.krige.score(         xy_train, y_train
    ))
    rk_test_score.append(m_rk.score(         X_test[:, reg_feature_ids], xy_test, y_test
    ))
    r_test_score.append(m_rk.regression_model.score(         X_test[:, reg_feature_ids], y_test
    ))
    k_test_score.append(m_rk.krige.score(         xy_test, y_test
    ))
rk_train_score = np.array(rk_train_score)
r_train_score = np.array(r_train_score)
k_train_score = np.array(k_train_score)
rk_test_score = np.array(rk_test_score)
r_test_score = np.array(r_test_score)
k_test_score = np.array(k_test_score)

print(rk_train_score.mean(), rk_train_score.std())
print(rk_test_score.mean(), rk_test_score.std())
# Regression coefficients
#print("\nRegression Intercept: {}".format(m_rk.regression_model.intercept_))
#print("Regression Coefficients: \n", m_rk.regression_model.coef_)
#print('\nRK score (R^2): ', m_rk.score( \
#        analysis['X'][:, reg_feature_ids], analysis['xy'], y_krig
#))
#print('Regression score (R^2): ', m_rk.regression_model.score( \
#        analysis['X'][:, reg_feature_ids], y_krig
#))
#print('Kriging score (R^2): ', m_rk.krige.score(analysis['xy'], y_krig))
m_rk.krige.model.display_variogram_model()

fig = plt.figure()
ax1 = plt.subplot(1, 1, 1)
ax1.plot(m_rk.krige_residual(xy_krig))
#ax2 = plt.subplot(2, 1, 2)

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(rk_train_score, 'k-o')
ax1.plot(r_train_score, 'b-o')
ax1.plot(k_train_score, 'r-o')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(rk_test_score, 'k--*')
ax2.plot(r_test_score, 'b--*')
ax2.plot(k_test_score, 'r--*')

print(m_rk.krige.model.lags)
print(rk_test_score.mean(), rk_test_score.std())

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(rk_train_score, 'k-o')
ax1.plot(r_train_score, 'b-o')
ax1.plot(k_train_score, 'r-o')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(rk_test_score, 'k--*')
ax2.plot(r_test_score, 'b--*')
ax2.plot(k_test_score, 'r--*')

# Make map for feature extraction and, optionally, plot data points on map
# Start by calculating the lower-left and upper-right corners of map.
# If they were provided as input above, then only adjust for the buffer width.
# Otherwise, calculate their values from the max and min of the data points
# In all cases, convert to numpy array at the end
if (len(latlon_ll) == 2) and (len(latlon_ll) == len(latlon_ur)):
    latlon_ll[0] -= latlon_buf[0]
    latlon_ll[1] -= latlon_buf[1]
    latlon_ur[0] += latlon_buf[0]
    latlon_ur[1] += latlon_buf[1]
else:
    latlon_ll = [             analysis[:,latlon_ids[0]].min() - latlon_buf[0],             analysis[:,latlon_ids[1]].min() - latlon_buf[1]
    ]
    latlon_ur = [             analysis[:,latlon_ids[0]].max() + latlon_buf[0],             analysis[:,latlon_ids[1]].max() + latlon_buf[1]
    ]
latlon_ll = np.array(latlon_ll)
latlon_ur = np.array(latlon_ur)

# Calculate x,y pairs for lower-left and upper-right map corners, then
# for housing data lat/lon pairs
x0, y0 = inputs['mapxy_CRS'].transform_point(latlon_ll[1], latlon_ll[0], inputs['geodetic_CRS'])
x1, y1 = inputs['mapxy_CRS'].transform_point(latlon_ur[1], latlon_ur[0], inputs['geodetic_CRS'])

# Set up WMTS service for satellite images
wmts = WebMapTileService(URL)

# Define Natural Earth Features to overlay on satellite images
RIVERS_10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',         edgecolor=cfeature.COLORS['water'], facecolor='none', linewidth=2
)
RIVERS_NA_10m = cfeature.NaturalEarthFeature('physical', 'rivers_north_america', '10m',         edgecolor=cfeature.COLORS['water'], facecolor='none', linewidth=1
)
LAKES_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='c', facecolor='c')
LAKES_NA_10m = cfeature.NaturalEarthFeature('physical', 'lakes_north_america', '10m',         edgecolor='c', facecolor='c', linewidth=1
)
OCEAN_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='b', facecolor='b')
MINOR_ISLANDS_COASTLINE_10m = cfeature.NaturalEarthFeature('physical', 'minor_islands_coastline',         '10m', edgecolor='k', facecolor='none'
)
COASTLINE_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='k',         facecolor='none'
)
BORDERS2_10m = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces',         '10m', edgecolor='grey', facecolor='none'
)

# Define figure and axis
# The following gives aspect ratio for Mercator projection (?)
figsize_x = 2 * figsize_y * (x1 - x0) / (y1 - y0)
fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=fig_dpi)
ax = plt.axes([0, 0, 1, 1], projection=inputs['mapxy_CRS'])
ax.set_xlim((x0, x1))
ax.set_ylim((y0, y1))

# Plot satellite images
ax.add_wmts(wmts, layers[0], wmts_kwargs={'time': date_str})

# Plot features: physical, cultural, etc., as defined above
ax.add_feature(OCEAN_10m, zorder=1)
ax.add_feature(LAKES_NA_10m, zorder=1)
ax.add_feature(LAKES_10m, zorder=1)
ax.add_feature(RIVERS_NA_10m, zorder=1)
ax.add_feature(RIVERS_10m, zorder=1)
ax.add_feature(BORDERS2_10m, zorder=2)
ax.add_feature(MINOR_ISLANDS_COASTLINE_10m, zorder=2)
ax.add_feature(COASTLINE_10m, zorder=2)

# Plot data points if plot_datapoints = True
if plot_datapoints:
    if plot_dptype == "scatter":
        ph = ax.scatter(analysis['X'][:, latlon_ids[1]], analysis['X'][:, latlon_ids[0]],                 c=analysis['y'], cmap=plot_dpcolormap,                 marker='o', edgecolor='none', facecolor='none',                 alpha=1, zorder=3, transform=inputs['geodetic_CRS']
        )
    elif plot_dptype == "hexbin":
        # Hexbins don't show up but no error???
        ph = ax.hexbin(analysis['X'][:, latlon_ids[1]], analysis['X'][:, latlon_ids[0]], mincnt=1,                 gridsize=20, C=analysis['y'], cmap=plot_dpcolormap, zorder=3,                 transform=inputs['geodetic_CRS']
        )

# Plot colorbar if data points were plotted and plot_dpcolorbar=True
if plot_datapoints and plot_dpcolorbar:
    plt.colorbar(ph, ax=ax)

# Save figure and close
if savefig_on:
    plt.savefig(savefig_name,dpi=fig_dpi, bbox_inches='tight')
plt.close()

# Image feature extraction parameters
# Define # of pixels from center pixel defined by data point lat/lon
# The actual buffer distance (in m) = impx_bufx*pixelXSize, impx_bufy*pixelYSize
dppx_bufxy = [10, 10]

# Load image that was just written to file and define its x/y grid
# Inspired in part by http://cgcooke.github.io/GDAL/
#img  = np.asarray(Image.open(savefig_name))
img = cv2.imread(imagefe_name)

# Define two lists with the x- and y-values of each pixel
yPixels, xPixels, nBand = img.shape  # number of pixels in y, x
pixelXSize =(x1-x0)/xPixels # size of the pixel in X direction     
pixelYSize = (y0-y1)/yPixels # size of the pixel in Y direction
pixelX = np.arange(x0, x1, pixelXSize)
pixelY = np.arange(y1, y0, pixelYSize)

# Identify image index corresponding to Cal housing lat/lon (analysis['xy'])
#ypi = 

# Plot histogram of entire map image in bgr
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr/(img.size/3),color = col)
    plt.xlim([-1,256])







