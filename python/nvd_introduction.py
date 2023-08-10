#Parsing using ElelmentTree
import xml.etree.ElementTree as ET
import csv

#getting all XML files in a python list
import glob
x = glob.glob("raw_data/*.xml")

#extarcting fields and writing them into a CSV file
yearRange =2002;
for y in x:
    CVE_tree = ET.parse(y)
    CVE_root= CVE_tree.getroot()
    f = open(str(yearRange)+".csv", 'w');
    yearRange= yearRange+1;
    CVE_count = 0;
    CVE_listOfId = [];
    for entry in CVE_root:
        cve_id = "";
        cwe_id = "";
        modified_date = "";
        cvss = "";
        for child in entry:
        #Print Child.tag will help you code further to identify child nodes
        #print (child.tag) 
            root = '{http://scap.nist.gov/schema/vulnerability/0.4}';
            childList = ['cve_id','published-datetime'];
            
            if (child.tag == '{http://scap.nist.gov/schema/vulnerability/0.4}cve-id'):
                cve_id = child.text;
            if (child.tag == '{http://scap.nist.gov/schema/vulnerability/0.4}cwe'):
                cwe_id = child.attrib['id'];        
            if (child.tag == '{http://scap.nist.gov/schema/vulnerability/0.4}summary'):
                modified_date = child.text;
            
    #Dont write header if you will be using the merged database
    #Head = "CVE ID,CWE ID,Timestamp\n";
    #f.write(Head);
        vuln = '{o1},{o2},{o3}\n'.format(o1=cve_id,o2=cwe_id,o3=modified_date);
        f.write(vuln);
        CVE_count = CVE_count +1;
    #print (CVE_count)
    f.close();
    

#merge all years into one
#only perform if data is required for all years
fout=open("Merged_2002-17.csv","a")
# first file:
Head = "CVE ID,CWE ID,Summary\n";
fout.write(Head);
for line in open("2002.csv"):
    fout.write(line)
# now the rest:    
for num in range(2003,2017):
    f = open(str(num)+".csv")
    #f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()

#using panda 
import pandas as pd
import csv

#We first load all file paths
import glob
nvd_filepaths = glob.glob("data/*.csv")
#Then we prepare a list, that will contain all the tables that exist in these file paths
nvd_dataframes = []
for nvd_filepath in nvd_filepaths:
    #the csvs do not contain headers, so are added here. TO-DO: Add headers to CSV when they are generated.
    nvd_dataframes.append(pd.read_csv(nvd_filepath,names=['cve_id', 'cwe_id','timestamp']))

#Choose the first dataframe at position 0
nvd_df = nvd_dataframes[0]
#Parse the timestamp field turning it into a datetimeindex object, and then access the month attribute
nvd_df['month'] = pd.DatetimeIndex(nvd_df['timestamp']).month

#Now that we have a month column, we can 'group by' the table by the month column. 
nvd_df = nvd_df.groupby(by=['month'])['cve_id','cwe_id'].count()
nvd_df

#All that is left is divide row-wise the number of cwe_ids, by the numter of cve_ids. 
#Since the cwe_ids are never null, then they effectively represent the number of rows for the given month. 
#cwe_id, instead, that can be null, will only counted when it occurs. 
#Dividing one by the other, gives us the cwe_coverage we desire for the timeseries.
nvd_df['cwe_coverage'] = nvd_df['cwe_id']/nvd_df['cve_id']
nvd_df

def calculate_cwe_coverage(nvd_df):
    #Parse the timestamp field turning it into a datetimeindex object, and then access the month attribute
    nvd_df['month'] = pd.DatetimeIndex(nvd_df['timestamp']).month
    #Now that we have a month column, we can 'group by' the table by the month column. 
    nvd_df = nvd_df.groupby(by=['month'])['cve_id','cwe_id'].count()
    nvd_df['cwe_coverage'] = nvd_df['cwe_id']/nvd_df['cve_id']
    return nvd_df
    

cwe_coverage_dfs = []
for nvd_df in nvd_dataframes: 
    cwe_coverage_dfs.append(calculate_cwe_coverage(nvd_df))
#cwe coverage for the 3rd dataset.
print(cwe_coverage_dfs)

#imports for histogram
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.io import output_notebook
from bokeh.charts import Bar
import matplotlib.pyplot as plot
from datetime import datetime
output_notebook() 

color_map = {
'2007': 'red',
'2008': 'green',
'2009': 'yellow',
'2010': 'violet',
'2011': 'indigo',
'2012': 'brown',
'2013': 'black',
'2014': 'blue',
'2015': 'orange',
'2016': 'olive',
'2017': 'navy',
}

def create_multi_line(vul):
    vul_plot = plot.subplot(111)
    map={}
    year = 2006;
    for frame in cwe_coverage_dfs:
        year+=1;
        map[year]= frame['cwe_coverage']
        vul_plot.plot(map[year], label = year )
    vul_plot.set_ylabel('cwe coverage')
    vul_plot.set_xlabel('month')
    vul_plot.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    vul_plot.set_title("Count of Vulnerability type per month")
    vul_plot.set_autoscaley_on(False)
    vul_plot.set_ylim([0,1])
    vul_plot.set_autoscalex_on(False)
    vul_plot.set_xlim([1,12])
    vul_plot.xaxis.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12])   
    plot.show()

create_multi_line(cwe_coverage_dfs)


