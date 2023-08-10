get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import numpy as np

jrm = models.JansenRit(mu=0., v0=6.)
monitor = monitors.TemporalAverage(period=2 ** -2)

phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max-jrm.p_min) * 0.5 )**2 / 2.
sigma         = numpy.zeros(6) 
sigma[3]      = phi_n_scaling

# the other aspects of the simulator are standard
sim = simulator.Simulator(
    model=jrm,
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.SigmoidalJansenRit(a=10.0),
    integrator=integrators.HeunStochastic(dt=2 ** -4, noise=noise.Additive(nsig=sigma)),
    monitors=[monitor],
    simulation_length=1e3,
).configure()

# run it
(time_array, data_array), = sim.run()

import uuid
import json
from datetime import datetime

BOOL_VALUE_PREFIX = "bool:"
DATETIME_VALUE_PREFIX = "datetime:"
DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def date2string(date_input, complex_format=True, date_format=None):
    """Convert date into string, after internal format"""
    if date_input is None:
        return "None"

    if date_format is not None:
        return date_input.strftime(date_format)

    if complex_format:
        return date_input.strftime(COMPLEX_TIME_FORMAT)
    return date_input.strftime(SIMPLE_TIME_FORMAT)


def serialize_value(value):
    """
    This method takes a value which will be stored as metadata and 
    apply some transformation if necessary
      
    :param value: value which is planned to be stored
    :returns:  value to be stored
     """
    if value is None:
        return ''
    # Force unicode strings to simple strings.
    if isinstance(value, unicode):
        return str(value)
    # Transform boolean to string and prefix it
    elif isinstance(value, bool):
        return BOOL_VALUE_PREFIX + str(value)
    # Transform date to string and append prefix
    elif isinstance(value, datetime):
        return DATETIME_VALUE_PREFIX + date2string(value, date_format=DATE_TIME_FORMAT)
    else:
        return json.dumps(value)
    
    
def generate_guid():
    """ 
    Generate new Global Unique Identifier.
    This identifier should be unique per each station, 
    and unique for different machines.
    """
    return str(uuid.uuid1())

import h5py
f = h5py.File("TimeSeriesRegion.h5", 'w')

series_of_time = time_series.TimeSeries(data=data_array, time=time_array, sample_period=monitor.period)
state_variable_dimension_name = series_of_time.labels_ordering[1]
selected_vois = [jrm.variables_of_interest[idx] for idx in monitor.voi]
series_of_time.labels_dimensions[state_variable_dimension_name] = selected_vois
series_of_time.configure()

time_set = f.create_dataset("time",data=series_of_time.time, maxshape=(None,))
time_set.attrs['TVB_Minimum'] = np.min(series_of_time.time)
time_set.attrs['TVB_Maximum'] = np.max(series_of_time.time)
time_set.attrs['TVB_Mean'] = np.mean(series_of_time.time)

data_set = f.create_dataset("data",data=series_of_time.data)
data_set.attrs['TVB_Minimum'] = np.min(series_of_time.data)
data_set.attrs['TVB_Maximum'] = np.max(series_of_time.data)
data_set.attrs['TVB_Mean'] = np.mean(series_of_time.data)

from tvb.basic.profile import TvbProfile

f.attrs['TVB_Connectivity'] = "f6be362b-5bb4-11e5-8b0d-a45e60e5b22f"
f.attrs['TVB_Create_date'] = serialize_value(datetime.now())
f.attrs['TVB_Data_version'] = TvbProfile.current.version.DATA_VERSION
f.attrs['TVB_Gid'] = generate_guid()
f.attrs['TVB_Has_surface_mapping'] = "true"
f.attrs['TVB_Has_volume_mapping'] = "false"
f.attrs['TVB_Invalid'] = serialize_value(False)
f.attrs['TVB_Is_nan'] = serialize_value(bool(np.isnan(data_array).any()))
f.attrs['TVB_Labels_dimensions'] = serialize_value(series_of_time.labels_dimensions)
f.attrs['TVB_Labels_ordering'] = serialize_value(series_of_time.labels_ordering)
f.attrs['TVB_Length_1d'] = serialize_value(series_of_time.length_1d)
f.attrs['TVB_Length_2d'] = serialize_value(series_of_time.length_2d)
f.attrs['TVB_Length_3d'] = serialize_value(series_of_time.length_3d)
f.attrs['TVB_Length_4d'] = serialize_value(series_of_time.length_4d)
f.attrs['TVB_Module'] = "tvb.datatypes.time_series"
f.attrs['TVB_Nr_dimensions'] = serialize_value(series_of_time.nr_dimensions)
f.attrs['TVB_Region_mapping'] = "002d1d23-5bb5-11e5-999a-a45e60e5b22f"
f.attrs['TVB_Sample_period'] = serialize_value(series_of_time.sample_period)
f.attrs['TVB_Sample_period_unit'] = serialize_value(series_of_time.sample_period_unit)
f.attrs['TVB_Sample_rate'] = serialize_value(series_of_time.sample_rate)
f.attrs['TVB_Start_time'] = serialize_value(series_of_time.start_time)
f.attrs['TVB_State'] = "INTERMEDIATE"
f.attrs['TVB_Subject'] = "FromIPython"
f.attrs['TVB_Title'] = serialize_value(series_of_time.title)
f.attrs['TVB_Type'] = "TimeSeriesRegion"
f.attrs['TVB_User_tag_1'] = "You can type any text you want here"
f.attrs['TVB_User_tag_2'] = ""
f.attrs['TVB_User_tag_3'] = ""
f.attrs['TVB_User_tag_4'] = ""
f.attrs['TVB_User_tag_5'] = ""
f.attrs['TVB_Visible'] = serialize_value(True)

f.close()



