#output_notebook function must be imported and called in order to output Bokeh figures within the notebook.  
from bokeh.io import output_notebook
output_notebook()

#Function to count all nodes that are direct children of Attack Pattern entries.  In the future, if there are other cases
#in which the important fields are children of a direct child, a special case should be added like one has been for the
#"Description" field.
def extract_label(node):
    if node.tag == 'Description':  
        for label in node:
            tag = label.tag
            if tag in frequencies:      # if the tag is already in the dictionary
                frequencies[tag] +=1    # add 1 to the count 
            else:
                frequencies[tag] = 1    # else, create an entry in the dictionary, starting the count at 1
    else:    #for all non-special cases
        if node.tag in frequencies:    # if the tag is already in the dictionary
            frequencies[node.tag] += 1 # add 1 to the count 
        else:
            frequencies[node.tag] = 1  # else, create an entry in the dictionary, starting the count at 1

            
import lxml.etree # LXML etree is used in place on Python's standard ElementTree.
tree = lxml.etree.parse('capec2.9.xml') # Outputs the results of the parsing into 'tree'
root = tree.getroot() # Grabs the root of the ElementTree element and places into 'root' [Both tree and root can be renamed]

# Remove namespaces from XML.  
for elem in root.getiterator(): 
    if not hasattr(elem.tag, 'find'): continue  # (1)
    i = elem.tag.find('}') # Counts the number of characters up to the '}' at the end of the XML namespace within the XML tag
    if i >= 0: 
        elem.tag = elem.tag[i+1:] # Starts the tag a character after the '}'

# Count fields using the previously defined extract_label function
frequencies = {}
for AttackPatternEntry in root[2]: # For each Attack Pattern Entry in the Attack Patterns Table (root[2])
    for Field in AttackPatternEntry: # For each Field in the current Attack Pattern entry
        extract_label(Field) 
        

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Range1d

data = {}  # Initialize dictionary 
data['Entries'] = frequencies # Place frequency count dictionary into the data dictionary.  More dictionaries may be added later
                              # to create stacked histograms with multiple sets of data.

df_data = pd.DataFrame(data).sort_values(by='Entries', ascending='true')  # Create dataframe from the data dictionary
series = df_data.loc[:,'Entries'] # Generate data series from the dataframe 

p = figure(width=800, y_range=series.index.tolist(), title="Attack Pattern Histogram") # Create figure 

#Set label details for the two axes
p.xaxis.axis_label = 'Frequency'
p.xaxis.axis_label_text_font_size = '10pt'
p.xaxis.major_label_text_font_size = '8pt'

p.yaxis.axis_label = 'Field'
p.yaxis.axis_label_text_font_size = '10pt'
p.yaxis.major_label_text_font_size = '8pt'


# Output horizontal histogram.  Bokeh function for this is currently being developed, and will be simpler to use in future.
j = 1 # Initializes count
for k,v in series.iteritems(): # k = key, v = value; these can be renamed more descriptively if needed.
  p.rect(x=v/2, y=j, width=abs(v), height=0.4, 
    width_units="data", height_units="data")
  j += 1
show(p)

#Counts co-occurence between field pairs - this code counts co-occurrence between all pairs of fields.  It currently also counts
#co-occurence between each field and itself, which can be ignored for now, but should be adjusted in the future.

co_occur = {}

for keys in frequencies:        #for each field in the frequencies dict
    co_occur[keys] = {}      #create a new dict inside the co-occurrence dict
    for key in frequencies:  #create a key in each individual field dict 
        co_occur[keys][key] = 0  #and set the values to 0

for fields in co_occur:           #for each dict in co_occur
    for test in co_occur[fields]:  #search each individual field in that dict
        for _ in root[2]:  #for each individual entry in the attack patterns table in the XML
            for column in _: #search each field in that entry
                if column.tag == fields: #if there is a field tag matching the current dict
                    for column in _: #for each field in the same entry
                        if column.tag == test: #if there is also a field tag matching the current field query
                            co_occur[fields][test] +=1 #add 1 to the count
#print(co_occur)

