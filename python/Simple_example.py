get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use(['seaborn-ticks', 'seaborn-talk'])

import numpy as np
import pandas as pd
import xarray as xr

fgm_ensemble = (
    xr.open_dataset("../data/fgm.all_cases.usa_subset.nc")
      .sel(pol='REF', ic=1, dec='1980-2010')
      .drop(['pol', 'ic', 'cs', 'dec', 'lev'])
      .load()
)
fgm_ensemble.info()

# ... remove the trend from -- all -- the data

# Select temperature data
t = fgm_ensemble['TEMP']
ds = t.mean(['lat', 'lon'])

# Re-sample annualy
ds_yearly = ds.resample(time='AS').mean()

# Convert to dataframe for plotting
df = ds_yearly.to_dataframe('test').reset_index()
df['idx'] = range(len(df))
print(df.head())

import seaborn as sns

sns.regplot('idx', 'test', df)

t = fgm_ensemble['TEMP']
ds = t.isel(lat=7, lon=7)
ms = MonthSelector(1, width=0)
ds = ms.fit_transform(ds)

(ds - ds.mean('time')).plot()

from stat_pm25.sklearn import *

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class PCRModel(DatasetModel):
    """ Performs principal component regression model on gridded data.

    This class uses the `DatasetModel` framework to implement a principal
    component regression - a linear regression on a set of features which has
    been pre-processed using principal component analysis. It only requires a
    single additional argument on top of `DatasetModel`, the number of
    components to retain.

    This example illustrates how you can very easily put together complex
    analyses and deploy them onto your gridded model output, all using very
    simple building blocks.

    """

    def __init__(self, *args, n_components=3, month=6, **kwargs):
        # If you need to add arguments, add them as named keyword arguments in
        # the appropriate place (as shown above), and set them first in the
        # method.
        self.n_components = n_components
        self.month = month
        
        # Modify any pre-set parameters, or hard-code them otherwise. For
        # instance, if you want to pre-process your data, this would be the
        # place to specify how to do so. Doing so here has the advantage that
        # you will be able to immediately apply your `predict()` method
        # to new data without pre-processing it - all that logic will be
        # saved

        # Zero out dilat and dilon, since we don't need to search around
        # neighboring grid cells
        self.dilat = self.dilon = 0

        # Set a pre-processor pipeline
        self.preprocessor = Pipeline([
            ('subset_time', MonthSelector(self.month, width=0)),
            # We didn't include the de-trending component below
            ('detrend', YearlyMovingAverageDetrender())
        ])
        
        # Call the parent superconstructor
        super().__init__(*args, **kwargs)

    def cell_kernel(self, gcf):
        """ Fit a model at a single grid cell.

        """

        # First, get the predictand data at the grid cell we care about. We
        # don't necessarily have to be super pedantic about this; we can just
        # use normal xarray selection methods if we want, although comments
        # below is how we could accomplish this using our specialized
        # Transformer classes
        # local_selector = DatasetSelector(
        #     sel='isel', lon=gcf.ilon, lat=gcf.ilat
        # )
        # y = local_selector.fit_transform(self.data[self.predictand])
        y = self.data[self.predictand].isel(lat=gcf.ilat, lon=gcf.ilon)

        # Prepare features timeseries. We want to fully include all the steps
        # to extract our features from the original, full dataset in here
        # so that our logic for re-applying the pipeline for prediction
        # later on will work similarly
        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
            ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('pca', PCA(n_components=self.n_components)),
            ('linear', LinearRegression()),
        ])

        # Fit the model/pipeline
        _model.fit(self.data, y)
        # Calculate some sort of score for archival
        _score = _model.score(self.data, y)
        # Encapsulate the result within a GridCellResukt
        gcr = GridCellResult(_model, self.predictand, self.predictors, 
                             _score, rand=np.random.randint(10), x=1)
        
        return gcr

predictand = 'PM25'
predictors = ['TEMP', 'RH', 'PRECIP', 'U', 'V']

train_data = fgm_ensemble.isel(time=slice(0, 12*25))
test_data = fgm_ensemble.isel(time=slice(12*25, None))

model = PCRModel(train_data, predictand, predictors, month=1,
                 verbose=True)
model.fit_parallel(3)

test_pm25 = model.predict(test_data, preprocess=True)

test_pm25.PM25.isel(time=0).plot.imshow()

score = model.score
score.plot.imshow()

rand = model.get_result_stat('x')['x']
rand.plot.imshow()

for gcr, gcf in model._gcr_gcf_iter:
    print(gcr)

import xarray as xr
import seaborn as sns

t = (
    xr.open_dataset("../data/fgm.all_cases.usa_subset.nc")
    ['TEMP']
    .sel(dec='1980-2010', pol='REF')
    .mean(['lat', 'lon'])
    .drop(['pol', 'cs', 'dec', 'lev'])
    .resample(time='AS').max('time')
)
t_df = t.to_dataframe().reset_index()
sns.tsplot(t_df, 'time', 'ic', value='TEMP', err_style='unit_traces')

ds = t.copy()

time = 'time'
aux_times = 'ic'

nt = len(ds[time])

# Loop over all the unit dim values
_dss = []
for i in range(len(ds[aux_times])):
    ds_aux_i = ds.isel(**{aux_times: i})
    
    # NOTE: Assume that our time offset will be in years, and that we can
    #       safely cast the existing timeseries to monthly values
    delta = np.timedelta64(nt*i, 'Y')
    ds_aux_i[time].values =         ds_aux_i[time].values.astype('datetime64[M]') + delta
        
    _dss.append(ds_aux_i)
    
ds_flat = xr.concat(_dss, time)

ds_flat.plot()

from stat_pm25.sklearn import *

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class NeighborhoodFeatures(TransformerMixin, NoFitMixin):
    """ Given a subset of a Dataset, extract a feature for
    each value along a given dimension.
    
    """
    
    def __init__(self, feature, dim='cell'):
        self.feature = feature
        self.dim = dim
    
    def transform(self, X):
        # Copy our dataset
        feats = X.copy().drop([self.feature, ])
        feat_names = []
        for i in range(len(X[self.dim])):
            _ds = X.isel(**{self.dim: i})
            feat_name = '{}_{:d}'.format(self.feature, i)
            print(feats)
            feats[feat_name] = _ds
        feats = feats[feat_names]
        
        return feats        

    
class LocalModel(DatasetModel):
    """ Performs a simple linear regression between local predictands
    and features generated from area-averaging around each cell as well
    as sampling from around that cell.

    """

    def __init__(self, *args, n_components=3,  **kwargs):
        self.n_components = n_components

        # This time, we *do* need dilat/dilon, since we need to be able
        # to search a radius around each cell. Let's choose a radius of 1,
        # which will search for one cell in each direction.
        self.dilat = 1
        self.dilon = 1

        # Set a pre-processor pipeline
        self.preprocessor = None
        
        # Call the parent superconstructor
        super().__init__(*args, **kwargs)

    def cell_kernel(self, gcf):
        """ Fit a model at a single grid cell.

        """
        # Select just the values at the local grid cell
        local_selector = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat)
        # Select the set of values within a grid cell's neighborhood
        area_selector = DatasetSelector(sel='isel',
                                        lon=gcf.ilon_range, lat=gcf.ilat_range)
        
        y = self.data[self.predictand].isel(lat=gcf.ilat, lon=gcf.ilon)

        # Select local features
        local_pipeline = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)),
            ('local_predictors', FieldExtractor(self.predictors)),
        ])
        
        # Select temperatures in an area around the local grid cell,
        # and average over them
        local_area_average_pipeline = Pipeline([
            ('get_temp', FieldExtractor(['TEMP', ])),
            ('get_local_area', area_selector),
            ('area_avg', DatasetFunctionTransformer('mean', 
                                                    args=(['lat', 'lon'],)
            )),
        ])
        
        # Select temperatures at the surrounding grid cells
        neighborhood_pipeline = Pipeline([
            ('get_temp', FieldExtractor(['TEMP', ])),
            ('get_local_area', area_selector),
            ('stack', Stacker(['lon', 'lat'], 'cell')),
            ('get_neighbors', NeighborhoodFeatures('TEMP', ))                        
        ])
        
        # Prepare features timeseries. We want to fully include all the steps
        # to extract our features from the original, full dataset in here
        # so that our logic for re-applying the pipeline for prediction
        # later on will work similarly
        _model = Pipeline([
            # Here we "branch" the pipeline to extract three sets of 
            # features to merge:
            ('make_features', DatasetFeatureUnion([
                # 1) Just the local predictors at the current grid cell
                ('local_features', local_pipeline),
                # 2) Our saved area average temperature pipeline
                # ('area_avg_feature', local_area_average_pipeline),
                # 3) Our saved neighborhood temperatures pipeline
                # ('neighborhood_features', neighborhood_pipeline)
                ('north', Pipeline([
                    ('get_temp_north', FieldExtractor(['TEMP', ])),
                    ('get_north', DatasetSelector(
                        sel='isel', lon=gcf.ilon, lat=gcf.ilat+1)),
                    ('rename_north', Renamer({'TEMP': 'TEMP_north'}))
                ])),
                ('south', Pipeline([
                    ('get_temp_south', FieldExtractor(['TEMP', ])),
                    ('get_south', DatasetSelector(
                        sel='isel', lon=gcf.ilon, lat=gcf.ilat-1)),
                    ('rename_south', Renamer({'TEMP': 'TEMP_south'}))
                ])),
            ])),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('pca', PCA(n_components=self.n_components)),
            ('linear', LinearRegression()),
        ])

        # Fit the model/pipeline
        _model.fit(self.data, y)
        # Calculate some sort of score for archival
        _score = _model.score(self.data, y)
        # Encapsulate the result within a GridCellResukt
        gcr = GridCellResult(_model, self.predictand, self.predictors, 
                             _score, rand=np.random.randint(10), x=1)
        
        return gcr

gcf = GridCellFactor(5, 5, 1, 1)
fe = FieldExtractor(['TEMP', ])
area_selector = DatasetSelector(sel='isel',
                                lon=gcf.ilon_range, lat=gcf.ilat_range)
pp = Pipeline([
    ('e', fe), ('a', area_selector),
    ('area_avg', DatasetFunctionTransformer('mean', 
                                            args=(['lat', 'lon'],)))
])
area_selector.transform(fgm_ensemble)
fe.transform(fgm_ensemble)
pp.transform(fgm_ensemble)

gcf = GridCellFactor(5, 5, 1, 1)
fe = FieldExtractor(['TEMP', ])
gs = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat-1)
rename = Renamer({'TEMP': 'TEMP_south'})

pp = Pipeline([
    ('a', fe), ('b', gs), ('c', rename)
])

pp.transform(fgm_ensemble)

predictand = 'PM25'
predictors = ['TEMP', 'RH', 'PRECIP', 'U', 'V']

train_data = fgm_ensemble.isel(time=slice(0, 12*25))
test_data = fgm_ensemble.isel(time=slice(12*25, None))

model = LocalModel(train_data, predictand, predictors,
                   # lat_range=[25, 50], 
                   lat_range=[30, 35],
                   lon_range=[-120, -75],
                   verbose=True)
model.fit_parallel(3)

rand = model.get_result_stat('rand')['rand']
rand.plot.imshow()



