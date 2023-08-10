db = 'stoqs_simz_aug2013'
survey = 'Dorado389_2013_225_01_225_01'
lopc = MeasuredParameter.objects.filter(dataarray__isnull=False).values(
    'measurement__instantpoint__timevalue', 'measurement__depth',
    'measurement__geom', 'parameter__domain', 'dataarray')

import pandas as pd
sep = pd.DataFrame.from_records(lopc.using(db).filter(
        measurement__instantpoint__activity__name__contains=survey,
        parameter__name='sepCountList'))

print len(sep)
sep[:2]

get_ipython().magic('matplotlib inline')
import pylab
pylab.rcParams['figure.figsize'] = (12.0, 4.0)
def label_plot(yaxis_name):
    pylab.title('LOPC data from ' + survey)
    pylab.ylabel(yaxis_name)
    pylab.xlabel('Size class (microns)')
    
for i,s in sep.iterrows():
    pylab.plot(s['parameter__domain'], s['dataarray'])

label_plot(Parameter.objects.using(db).get(name='sepCountList').long_name)

label_plot(Parameter.objects.using(db).get(name='sepCountList').long_name)
for i,s in sep.iterrows():
    pylab.plot(s['parameter__domain'][5:30], s['dataarray'][5:30])

label_plot(Parameter.objects.using(db).get(name='mepCountList').long_name)
mep = pd.DataFrame.from_records(lopc.using(db).filter(
        measurement__instantpoint__activity__name__contains=survey,
        parameter__name='mepCountList'))
for i,m in mep.iterrows():
    pylab.plot(m['parameter__domain'], m['dataarray'])

label_plot(Parameter.objects.using(db).get(name='mepCountList').long_name)
for i,m in mep.iterrows():
    pylab.plot(m['parameter__domain'][:70], m['dataarray'][:70])

# Create new Parameters in the database
p,created = Parameter.objects.using(db).get_or_create(
        name='lopc_total_count', units='count',
        long_name='Total of SEP and MEP Counts')
logp,created = Parameter.objects.using(db).get_or_create(
        name='log_lopc_total_count', units='log_count',
        long_name='Log of Total of SEP and MEP Counts')


# Assign new parameters to the ParameterGroup 'Measured in situ'
for px in [p, logp]:
    ParameterGroupParameter.objects.using(db).get_or_create(
            parametergroup=ParameterGroup.objects.using(db).get(
            name='Measured in situ'), parameter=px)

# Construct 3 query sets
mps = MeasuredParameter.objects.using(db).filter(dataarray__isnull=False,
        measurement__instantpoint__activity__name__contains=survey).order_by(
        'measurement__instantpoint__timevalue')
seps = mps.filter(parameter__name='sepCountList').values_list('dataarray', flat=True)
meps = mps.filter(parameter__name='mepCountList').values_list('dataarray', flat=True)

# Create new summarized MeasuredParameters
import math
for mp,sep,mep in zip(mps.filter(parameter__name='sepCountList'), seps, meps):
    MeasuredParameter.objects.using(db).get_or_create(parameter=p,
            measurement=mp.measurement, datavalue = sum(sep) + sum(mep))
    MeasuredParameter.objects.using(db).get_or_create(parameter=logp,
            measurement=mp.measurement, datavalue = math.log(sum(sep) + sum(mep), 10))

# Use core STOQS loader software to update the database with descriptive statistics
from loaders import STOQS_Loader
STOQS_Loader.update_ap_stats(db, activity=Activity.objects.using(db
                ).get(name__contains=survey), parameters=[p, logp])

# Report count of records loaded
print("Number of records loaded: {:d}".format(
        MeasuredParameter.objects.using(db).filter(parameter__name='lopc_total_count'
                                                  ).count() +
        MeasuredParameter.objects.using(db).filter(parameter__name='log_lopc_total_count'
                                                  ).count()))

from IPython.display import Image
Image('../../../doc/Screenshots/Screen_Shot_2015-10-03_at_3.38.57_PM.png')

samples = {}
for s in Sample.objects.using(db).select_related('instantpoint').filter(
    instantpoint__activity__name__contains=survey, sampletype__name='Gulper'):
    samples[int(s.name)] = s.instantpoint.timevalue

from datetime import timedelta
binning_secs = 20
label_plot(Parameter.objects.using(db).get(name='sepCountList').long_name)
for sa,tv in samples.iteritems():    
    trange = [tv - timedelta(seconds=binning_secs), tv + timedelta(seconds=binning_secs)]
    sep = pd.DataFrame.from_records(lopc.using(db).filter(
            measurement__instantpoint__timevalue__range = trange,
            parameter__name='sepCountList'))

    for i,se in sep.iterrows():
        pylab.plot(se['parameter__domain'][:70], se['dataarray'][:70],
                    label='Gulper {:d}'.format(sa))
        
pylab.legend()

label_plot(Parameter.objects.using(db).get(name='mepCountList').long_name)
for sa,tv in samples.iteritems():    
    trange = [tv - timedelta(seconds=binning_secs), tv + timedelta(seconds=binning_secs)]
    mep = pd.DataFrame.from_records(lopc.using(db).filter(
            measurement__instantpoint__timevalue__range = trange,
            parameter__name='mepCountList'))
    
    for i,me in mep.iterrows():
        pylab.plot(me['parameter__domain'][:70], me['dataarray'][:70],
                    label='Gulper {:d}'.format(sa))
        
pylab.legend()



