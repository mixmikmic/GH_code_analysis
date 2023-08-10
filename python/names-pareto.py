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

from matplotlib import pyplot as plt
from pandas.io import gbq
import pandas as pd

q = '''#standardSQL
SELECT
  name,
  name_total,
  SUM(name_total) OVER(ORDER BY name ASC) AS name_cumulative
FROM (
  SELECT
    name,
    SUM(number) AS name_total
  FROM
    `bigquery-public-data.usa_names.usa_1910_2013`
  GROUP BY
    name )
ORDER BY
  name ASC'''
df = gbq.read_gbq(q, project_id='swast-scratch', dialect='standard')

# Add a column that converts the cumulative total to percentages
total_names = df.name_cumulative.tail(1).values
df = df.assign(name_percent=pd.Series((df.name_cumulative * 100.0) / total_names).values)

def pareto(df, nletters):
    dff = df.groupby(by=lambda x: df.name[x][:nletters])
    dff = dff.agg({'name_total': 'sum', 'name_percent': 'max'})

    # Make a pareto plot (two y-axes)
    dff.name_total.plot.bar()
    return dff.name_percent.plot(secondary_y=True)

pareto(df, 1)

j = df.select(lambda x: df.name.values[x].startswith("J"))
pareto(j, 2)

jo = df.select(lambda x: df.name.values[x].startswith("Jo"))
pareto(jo, 3)

jor = df.select(lambda x: df.name.values[x].startswith("Jor"))
jor



