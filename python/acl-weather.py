# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime

from matplotlib import pyplot as plt
from pandas.io import gbq
import pandas as pd

q = '''#standardSQL
SELECT
  CAST(FORMAT('%s-%s-%s', year, mo, da) AS DATE) `date`,
  `min`,
  `temp`,
  `max`
FROM
  `bigquery-public-data.noaa_gsod.gsod20*`
WHERE
  stn = '722544'
  AND `min` != 9999.9
  AND `max` != 9999.9
  AND `temp` != 9999.9
  AND _TABLE_SUFFIX BETWEEN '06' and '16'
ORDER BY `date` ASC;'''
df = gbq.read_gbq(q, project_id='swast-scratch', dialect='standard')

df = df.assign(day_of_year=df.date.map(lambda dt : datetime.datetime.strptime(dt, '%Y-%m-%d').timetuple().tm_yday))

alpha = 0.5
ax = df.plot(kind='scatter', x='day_of_year', y='temp', alpha=alpha, color='gray');
df.plot(kind='scatter', x='day_of_year', y='min', color='blue', alpha=alpha, ax=ax);
df.plot(kind='scatter', x='day_of_year', y='max', alpha=alpha, color='red', ax=ax);
ax.get_figure().set_size_inches(14, 10)
ax.set_xlim(0, 366)

ax.axvline(x=datetime.datetime(2006, 9, 15).timetuple().tm_yday, color='k')
ax.axvline(x=datetime.datetime(2010, 10, 8).timetuple().tm_yday, color='b')
ax.set_xlim(160, 366)
ax.get_figure()



