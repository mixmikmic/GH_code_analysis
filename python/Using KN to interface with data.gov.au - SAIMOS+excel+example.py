import pandas, xlrd, requests, json
from pandas import np
import matplotlib.pyplot as plt
pandas.set_option('display.max_columns', 500)
get_ipython().magic('matplotlib inline')

response = requests.get("http://kn.csiro.au/api/dataset?id=http%3A%2F%2Foznome.csiro.au%2Fid%2Fdata%2Fdata-gov-au%2Fsaimos-biological-and-flow-cytometry-data-collected-from-ctd-stations-in-south-australia-i-20142")
json_data = response.json()

json_data

import uuid
from IPython.display import display_javascript, display_html, display
import json

class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict) or isinstance(json_data, list):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
        document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)

RenderJSON(json_data)

RenderJSON(json_data['resources'])

url = [resource for resource in json_data["resources"] if "Picophytoplankton" in resource["name"]][0]["url"]
url

r = requests.get(url)
book = xlrd.open_workbook(file_contents=r.content)

book.sheet_names()

dataframe = pandas.read_excel(url, sheetname='Converted_CLEAN')
dataframe.columns

dataframe.describe(include='all')

from FilteringWidget import FilteringWidget

filtered = FilteringWidget(dataframe, ['Station', 'Depth (m)'])

filtered.dataframe

filtered_frame = filtered.dataframe.replace('-', np.nan)

filtered_frame = filtered_frame[[ 'Synechococcus ','Prochlorococus', 'Picoeukaryotes', 'Rep', 'Depth (category)']]

filtered_frame

filtered_frame = filtered_frame.loc[filtered_frame['Rep'] == 2]

filtered_frame

filtered_frame.pop('Rep');

# Give the index a name
filtered_frame.index.name = 'experiment'

# Create a standard column from the current index
filtered_frame.reset_index(level=0, inplace=True)

# Create the multi-index from named columns
filtered_frame.set_index(['experiment', 'Depth (category)'], inplace=True)

filtered_frame.plot(kind='bar', stacked=True);

filtered_frame[['SAM8' in x[0] for x in filtered_frame.index]].plot(kind='bar', title='SAM8SG');

filtered_frame[['Surface' in x[1] for x in filtered_frame.index]].plot(kind='bar', title='Surface');

