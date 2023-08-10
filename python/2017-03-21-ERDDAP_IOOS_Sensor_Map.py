import requests
try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode


def encode_erddap(urlbase, fname, columns, params):
    """
    urlbase: the base string for the endpoint
             (e.g.: https://erddap.axiomdatascience.com/erddap/tabledap).
    fname: the data source (e.g.: `sensor_service`) and the response (e.g.: `.csvp` for CSV).
    columns: the columns of the return table.
    params: the parameters for the query.

    Returns a valid ERDDAP endpoint.
    """
    urlbase = urlbase.rstrip('/')
    if not urlbase.lower().startswith(('http:', 'https:')):
        msg = 'Expected valid URL but got {}'.format
        raise ValueError(msg(urlbase))

    columns = ','.join(columns)
    params = urlencode(params)
    endpoint = '{urlbase}/{fname}?{columns}&{params}'.format

    url = endpoint(urlbase=urlbase, fname=fname,
                   columns=columns, params=params)
    r = requests.get(url)
    r.raise_for_status()
    return url

try:
    from urllib.parse import unquote
except ImportError:
    from urllib2 import unquote


urlbase = 'https://erddap.axiomdatascience.com/erddap/tabledap'

fname = 'sensor_service.csvp'

columns = ('time',
           'value',
           'station',
           'longitude',
           'latitude',
           'parameter',
           'unit',
           'depth')
params = {
    # Inequalities do not exist in HTTP parameters,
    # so we need to hardcode the `>` in the time key to get a '>='.
    # Note that a '>' or '<' cannot be encoded with `urlencode`, only `>=` and `<=`.
    'time>': '2017-01-00T00:00:00Z',
    'station': '"urn:ioos:station:wmo:44011"',
    'parameter': '"Significant Wave Height"',
    'unit': '"m"',
}

url = encode_erddap(urlbase, fname, columns, params)

print(unquote(url))

from pandas import read_csv

df = read_csv(url, index_col=0, parse_dates=True)

# Prevent :station: from turning into an emoji in the webpage.
df['station'] = df.station.str.split(':').str.join('_')

df.head()

get_ipython().magic('matplotlib inline')

ax = df['value'].plot(figsize=(11, 2.75), title=df['parameter'][0])

params.update(
    {
        'value>': 6,
        'time>': '2016-01-00T00:00:00Z',
    }
)

url = encode_erddap(urlbase, fname, columns, params)

df = read_csv(url, index_col=0, parse_dates=True)

# Prevent :station: from turning into an emoji in the webpage.
df['station'] = df.station.str.split(':').str.join('_')

df.head()

def key(x):
    return x.month


grouped = df['value'].groupby(key)

ax = grouped.count().plot.bar()
ax.set_ylabel('Significant Wave Height events > 6 meters')
m = ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec'])

from IPython.display import HTML


fname = 'sensor_service.htmlTable'

params = {
    'time>': 'now-2hours',
    'time<': 'now',
    'station': '"urn:ioos:station:nerrs:wqbchmet"',
    'parameter': '"Wind Speed"',
    'unit': '"m.s-1"'
}

url = encode_erddap(urlbase, fname, columns, params)

iframe = '<iframe src="{src}" width="650" height="370"></iframe>'.format
HTML(iframe(src=url))

fname = 'sensor_service.png'

params = {
    'time>': 'now-7days',
    'station': '"urn:ioos:station:wmo:44011"',
    'parameter': '"Water Temperature"',
    'unit': '"degree_Celsius"',
}


width, height = 450, 500
params.update(
    {'.size': '{}|{}'.format(width, height)}
)

url = encode_erddap(urlbase, fname, columns, params)

iframe = '<iframe src="{src}" width="{width}" height="{height}"></iframe>'.format
HTML(iframe(src=url, width=width+5, height=height+5))

