get_ipython().magic('matplotlib inline')
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#import mpld3
from IPython.display import HTML
import folium

# IPython "version_information" extension needs to be installed
# %install_ext http://raw.github.com/jrjohansson/version_information/master/version_information.py
get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, pandas, matplotlib, mpld3, folium')

# For Folium map
def inline_map(map, height=500):
    """
    Folium inline, interactive map using Leaflet.
    Embeds the HTML source of the map directly into the IPython notebook.
    
    This method will not work if the map depends on any files (json data). Also this uses
    the HTML5 srcdoc attribute, which may not be supported in all browsers.
    """
    map._build_map()
    iframe = '<iframe srcdoc="{srcdoc}" style="width: 100%%; height: %dpx; border: none"></iframe>' % height
    return HTML(iframe.format(srcdoc=map.HTML.replace('"', '&quot;')))

# For matplotlib plots, including time series plots
matplotlib.style.use('ggplot')
#mpld3.enable_notebook()

create_ylabel = lambda indf: indf.ix[0].UnitsTypeCV + ' (' + indf.ix[0].UnitsAbbrev + ')'

def get_title(VariableCode, observations, SiteRelatedSF, lat, lon):
    title = "%s (%s) specimen measurement results\n at %s (lat %.3f, lon %.3f), a %s / %s Site " % (
        VariableCode, observations[0]['Variable']['VariableNameCV'],
        SiteRelatedSF['SamplingFeatureName'], lat, lon,
        SiteRelatedSF['SamplingFeatureTypeCV'], SiteRelatedSF['Site']['SiteTypeCV']
    )
    return title

# For ODM2 REST requests
params = {'format': 'json'}

def odm2rest_request(service, request, value, params):
    baseurl = "http://sis-devel.cloudapp.net:80"
    request_url = "%s/%s/%s/%s" % (baseurl, service, request, value)
    r = requests.get(request_url, params=params)
    return r.json()

def get_odm2rest_measresultvalues_df(obs, params):
    """
    Issue ODM2 REST Observatoins/value request, return a selective "record" (dict)
    """
    resultvalues_obs = odm2rest_request("Observations", "value", obs['ResultUUID'], params)
    # For measurement result type, there's only one result value per result
    measresult = resultvalues_obs[0]
    measresultvalues = measresult['MeasurementResult']['MeasurementResultValues']
    # Note that the last 3 dict items below are actually by Result, not by ResultValue
    resultvalue_record = {'ValueDateTime': measresultvalues['ValueDateTime'],
                          'DataValue': measresultvalues['DataValue'],
                          'CensorCodeCV': measresult['MeasurementResult']['CensorCodeCV'],
                          'UnitsTypeCV': obs['Unit']['UnitsTypeCV'],
                          'UnitsAbbrev': obs['Unit']['UnitsAbbreviation']
                         }
    return resultvalue_record

VariableCode = 'FPOC'

observations = odm2rest_request("Observations", "variableCode", VariableCode, params)
len(observations)

# ResultDateTime range (min & max)
ls = pd.to_datetime([obs['ResultDateTime'] for obs in observations])
ls.min(), ls.max()

# Related Site SamplingFeature (just one)
# ** Applicable also to all other specimens in the Marchantaria use case; for now **
SiteRelatedSF = observations[0]['RelatedFeatures'][0]
lat, lon = SiteRelatedSF['Site']['Latitude'], SiteRelatedSF['Site']['Longitude']

# Text/HTML content for marker popup
popup_str = '<b>%s</b><br>' % SiteRelatedSF['SamplingFeatureName']
popup_str += 'A %s / %s Site<br>' % (SiteRelatedSF['SamplingFeatureTypeCV'],
                                    SiteRelatedSF['Site']['SiteTypeCV'])
popup_str += 'SamplingFeatureUUID:<br>%s' % SiteRelatedSF['SamplingFeatureUUID']

mapheight = 300
map = folium.Map(width=500, height=mapheight, location=[lat, lon], zoom_start=6)
map.simple_marker([lat, lon], popup=popup_str)
inline_map(map, mapheight+10)

resultvalue_records = [get_odm2rest_measresultvalues_df(obs, params) for obs in observations]
len(resultvalue_records)

resultvalue_df = pd.DataFrame.from_records(resultvalue_records)
resultvalue_df['dtutc'] = pd.to_datetime(resultvalue_df['ValueDateTime'], 
                                         utc=True, infer_datetime_format=True)

resultvalue_df.head(10)

resultvalue_df.groupby(['UnitsTypeCV', 'UnitsAbbrev']).dtutc.nunique()

plt.figure(figsize=(13,6))

df = resultvalue_df[resultvalue_df.UnitsTypeCV == 'Mass concentration']
df.set_index('dtutc', inplace=True)
ax = df['DataValue'].plot(style='o-', label=df.ix[0].UnitsTypeCV)
ax.set_ylabel(create_ylabel(df))

df = resultvalue_df[resultvalue_df.UnitsTypeCV == 'Mass fraction']
df.set_index('dtutc', inplace=True)
df['DataValue'].plot(style='^-', label=df.ix[0].UnitsTypeCV,
                     ax=ax, secondary_y=True)
ax.right_ax.set_ylabel(create_ylabel(df))

ax.legend(loc=2)
ax.right_ax.legend(loc=1)
ax.xaxis.grid(True, which="major")
ax.set_xlabel('')
ax.set_title(get_title(VariableCode, observations, SiteRelatedSF, lat, lon));

VariableCode = 'NO3'
observations = odm2rest_request("Observations", "variableCode", VariableCode, params)
len(observations)

# ResultDateTime range (min & max)
ls = pd.to_datetime([obs['ResultDateTime'] for obs in observations])
ls.min(), ls.max()

resultvalue_records = [get_odm2rest_measresultvalues_df(obs, params) for obs in observations]
len(resultvalue_records)

resultvalue_df = pd.DataFrame.from_records(resultvalue_records)
resultvalue_df['dtutc'] = pd.to_datetime(resultvalue_df['ValueDateTime'], 
                                         utc=True, infer_datetime_format=True)

resultvalue_df.groupby(['UnitsTypeCV', 'UnitsAbbrev']).dtutc.nunique()

plt.figure(figsize=(13,6))

df = resultvalue_df
df.set_index('dtutc', inplace=True)
ax = df['DataValue'].plot(style='o-', label=df.ix[0].UnitsTypeCV)
ax.set_ylabel(create_ylabel(df))

ax.legend(loc=2)
ax.xaxis.grid(True, which="major")
ax.set_xlabel('')
ax.set_title(get_title(VariableCode, observations, SiteRelatedSF, lat, lon));

df['dayofyear'] = df.index.dayofyear

ax = df.plot(kind='scatter', x='dayofyear', y='DataValue', 
             xlim=[0,366], figsize=(5,5))
ax.set_ylabel(create_ylabel(df))
ax.set_title(VariableCode);

df['year'] = df.index.year

ax = df.plot(kind='scatter', x='dayofyear', y='DataValue', c='year', 
             s=40, colormap='Greens', xlim=[0,366], figsize=(6,6))
ax.set_ylabel(create_ylabel(df))
ax.set_title(VariableCode);



