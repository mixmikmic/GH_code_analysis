import pandas as pd
from pixiedust.display.app import *

df = pixiedust.sampleData(6, forcePandas=True)

# -------------------------------------------------------------------------------
# Copyright IBM Corp. 2018
# 
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------

@PixieApp
@Logger()
class Stats():
    
    def percentInDeviation(self, col_idx, num_devs=1):
        col_name = self.cols[col_idx]
        mean = self.descriptions[col_idx]['mean']
        std_dev = self.descriptions[col_idx]['std']
        minval = mean - (std_dev * num_devs)
        maxval = mean + (std_dev * num_devs)
        valsin = self.df[col_name][(self.df[col_name] > minval) & (self.df[col_name] < maxval)]
        pctin = (valsin.shape[0]/self.df[col_name].shape[0]*100)
        return pctin
    
    def makeDFOutsideDeviation(self, col_idx, num_devs=1):
        col_name = self.cols[col_idx]
        mean = self.descriptions[col_idx]['mean']
        std_dev = self.descriptions[col_idx]['std']
        minval = mean - (std_dev * num_devs)
        maxval = mean + (std_dev * num_devs)
        rowsout = self.df[(self.df[col_name] < minval) | (self.df[col_name] > maxval)]
        return rowsout
        
    def setup(self):
        self.df = self.pixieapp_entity
        self.cols = self.df.columns
        self.datatypes = []
        self.valuecounts = []
        # results of describing the column
        self.descriptions = []
        # 2d array: for nums, idx 1=%vals within 1 std of mean, idx 2=%vals within 2stds of mean, 0,0 for string or booleans
        self.means = [] 
                
        self.idxs = []
        i = 0
        for col in self.cols:
            self.idxs.append(i)
            
            description = self.df[col].describe()
            self.descriptions.append(description)
            vc = df[col].value_counts().count()
            self.valuecounts.append(vc)
            if vc == 2:
                self.datatypes.append('boolean-like')
                self.means.append([0, 0])
            elif df[col].dtype == 'object':
                self.datatypes.append('string')
                self.means.append([0, 0])
            else:
                self.datatypes.append(df[col].dtype)
                self.means.append([self.percentInDeviation(i,1), self.percentInDeviation(i,2)])
            i = i+1
    
    @route()
    @templateArgs
    def main_screen(self): 
        return """
        <div id="stats">
        <div id="stats-title-{{prefix}}"></div>
        <div id="main-screen-{{prefix}}">

        <style>
        .rendered_html td { vertical-align: bottom; color: #888888; padding: 8px 4px }
        .rendered_html td.fieldnames { text-align: left }
        .rendered_html td.name, .name { color: black; }
        .bignum { font-weight: 500; font-size: 200%; line-height: 2px}
        .warning { color: #c99a00; }
        </style>

        <table>
        <tbody>
        {% for i in this.idxs %}
        <tr>
          <td><button class="btn btn-default btn-xs">Select
            <target pd_target="stats-title-{{prefix}}" pd_options="title_text={{this.cols[i]}}" />
            <target pd_target="main-screen-{{prefix}}" pd_options="col_index={{i}}" />
            <target pd_target="charts-{{prefix}}" pd_entity pd_options="col_index={{i}};handlerId=histogram;valueFields={{this.cols[i]}};bins={{this.df[this.cols[i]].shape[0]/20}};rowCount=5000;chartsize=40;legend=false" />
          </button><td>
          <td class="fieldnames bignum name">{{this.cols[i]}}<td>
          <td class="datatypes">{{this.datatypes[i]}}<td>
          {% if this.datatypes[i]|string == "string" %}
            <td>--</td>
            <td class="name">unique:</td>
            <td><span class="bignum">{{this.descriptions[i]['unique']}}</span></td>
            <td colspan=6>  </td>
          {% elif (this.datatypes[i]|string).startswith("boolean") %}
            <td>{{this.datatypes[i]|string}}</td>
            <td colspan=6>  </td>
          {% else %}
            <td><span class="name">min: </span>{{this.descriptions[i]['min']}}<br/>
                <span class="name">max: </span>{{this.descriptions[i]['max']}}</td>
            <td class="name">mean:</td>
            <td><span class="bignum">{{this.descriptions[i]['mean']|round(2)}}</span></td>
            <td class="name">standard<br/>deviation</td>    
            <td><span class="bignum">{{this.descriptions[i]['std']|round(2)}}</span></td>
            <td class="name">% within<br/>1 std</td>
            <td><span class="bignum{% if this.means[i][0] < 68 %} warning{%endif%}">{{this.means[i][0]|round(0)}}</span></td>
            <td class="name">% within<br/>2 std</td>    
            <td><span class="bignum{% if this.means[i][1] < 95 %} warning{%endif%}">{{this.means[i][1]|round(0)}}</span></td>
          {%endif%}
        </tr>
        {% endfor %}
        </tbody>
        </table>
        </div>
        
        </div>
        """
    @route(title_text="*")
    def do_title(self, title_text):
        return """
        <style>.back-nav {float: right}</style>
        <div>
          <div class="back-nav"><button pd_options="showall=yes" class="btn btn-info">Back</button></div>
          <h1>{{title_text}}</h1>
        </div>
        """

    @route(col_index="*")
    @templateArgs
    def column_stats(self, col_index):
        i = int(col_index)
        description = self.descriptions[i]
        self.outliersDF = self.makeDFOutsideDeviation(i, 2)
#         self.outliersDF.reset_index(inplace=True)
        
        return """
        <div id="main-screen-{{prefix}}">
        <style>
        .rendered_html td { vertical-align: bottom; color: #888888; padding: 8px 12px }
        .rendered_html td.fieldnames { text-align: left }
        .rendered_html td.name, .name { color: black; }
        .bignum { font-weight: 500; font-size: 200%; line-height: 2px}
        .warning { color: #c99a00; }
        .rendered_html pre.code { font-size: 150%; padding: 8px 0; color: #888888}
        .rendered_html tbody tr:nth-child(odd) { background-color: #FFFFFF}
        #charts-{{prefix}} img { height: 84px;}
        </style>

        <table>
        <tbody>
        <tr>
          <td rowspan="2"><div id="charts-{{prefix}}"><h2>chart</h2></div></td>
          <td class="datatypes"><span class="name">type: </span>{{this.datatypes[i]}}<br/>
              <span class="name">count: </span>{{this.df[this.cols[i]].shape[0]}}<td>
          {% if this.datatypes[i]|string != "string" and not (this.datatypes[i]|string).startswith("boolean") %}
            <td class="name">mean:</td>
            <td><span class="bignum">{{this.descriptions[i]['mean']|round(2)}}</span></td>
            <td class="name">standard<br/>deviation</td>    
            <td><span class="bignum">{{this.descriptions[i]['std']|round(2)}}</span></td>
          {%endif%}
        </tr>

        <tr>
          {% if this.datatypes[i]|string == "string" %}
            <td>--</td>
            <td class="name">unique:</td>
            <td><span class="bignum">{{this.descriptions[i]['unique']}}</span></td>
            <td colspan=6>  </td>
          {% elif (this.datatypes[i]|string).startswith("boolean") %}
            <td>{{this.datatypes[i]|string}}</td>
            <td colspan=6>  </td>
          {% else %}
            <td><span class="name">min: </span>{{this.descriptions[i]['min']}}<br/>
                <span class="name">max: </span>{{this.descriptions[i]['max']}}</td>
            <td> </td>
          {%endif%}
          <td class="name">% within<br/>1 std</td>
          <td><span class="bignum{% if this.means[i][0] < 68 %} warning{%endif%}">{{this.means[i][0]|round(0)}}</span></td>
          <td class="name">% within<br/>2 std</td>    
          <td><span class="bignum{% if this.means[i][1] < 95 %} warning{%endif%}">{{this.means[i][1]|round(0)}}</span></td>
          <td><span class="name">Outliers</span></td>
          <td>
            <button pd_options="handlerId=dataFrame;maxRows={{this.outliersDF.shape[0]}};table_noschema=true" pd_entity=outliersDF pd_target="outliers-{{prefix}}" class="btn btn-warning btn-xs">Table</button> 
            <button pd_options="code_type=outliers" pd_target="outliers-{{prefix}}" class="btn btn-warning btn-xs">Selection code</button>
          </td>
        </tr>

        </tbody>
        </table>
        
        <div id="outliers-{{prefix}}"></div>
        <div id="outliers-{{prefix}}"></div>
        </div>
        """
    
    @route(code_type="*")
    @templateArgs
    def outliers_code(self, code_type):   
        idxs = ','.join(map(str, self.outliersDF.index.values))
        the_code = "newdf = df.loc[~df.index.isin([{0}])]".format(idxs)
        return """
        <div>
          <h3>Create a new DataFrame excluding these rows:</h3>
          <pre class="code">{{the_code}}</pre>
          <button class="btn btn-info" pd_script="get_ipython().set_next_input('{{the_code}}')">Write it</button>
        </div>
        """

stats = Stats()
stats.run(df, runInDialog='false')

