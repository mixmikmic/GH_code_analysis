#!/usr/bin/env python

# make sure to install these packages before running:
# pip install pandas
# pip install bokeh

import numpy as np
import pandas as pd
import datetime
import urllib

from bokeh.plotting import *
from bokeh.models import HoverTool
from collections import OrderedDict

query = ("https://data.phila.gov/resource/4t9v-rppq.json?$where=requested_datetime%20between%20%272016-09-01T00:00:00%27%20and%20%272016-10-01T00:00:00%27")
raw_data = pd.read_json(query, convert_dates=['expected_datetime','requested_datetime','updated_datetime'])

raw_data=raw_data.set_index(['service_request_id'])
raw_data.head()

