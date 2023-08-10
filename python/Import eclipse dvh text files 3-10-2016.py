get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Robin Cole' -u -d -v -p numpy,pandas,matplotlib")

import numpy as np
import itertools
import json
import re

file = 'case1_AAA.txt'
file_header =  open(file, 'r').readlines()[:15]
file_body =    open(file, 'r').readlines()[16:]  # first 16 lines are header, for some reason was not opening header

plan = {}   # to contain all file plan info

# parse the header here
for line in file_header:
    if line.replace(u'\ufeff', '').startswith('Patient Name'):
        plan['Patient Name'] = line.replace(u'\ufeff', '').split(':')[1].rstrip()
    elif line.startswith('Patient ID'):
        plan['Patient ID'] = line.split(':')[1].rstrip() 
    elif line.startswith('Plan:'):
        plan['Plan'] = line.split(':')[1].rstrip() 
    elif line.startswith('Prescribed dose [Gy]'):
        plan['Prescribed dose [Gy]'] = line.split(':')[1].rstrip()
    else:
        pass

#############################

## get all structures
Structures = []   # create list 
print(len(file_body))
for i, line in enumerate(file_body): 
    matchObj = re.match( r'^Structure: (.*?$)', line, re.M|re.X)
    if matchObj:
        Structures.append(matchObj.group(1))
plan['Structures'] = Structures  

#############################

plan

i = 0
for key, Structure_group in itertools.groupby(file_body, lambda line: line.startswith('Structure:')):  # group on Structure
    print(i)    
    for key, sub_group in itertools.groupby(Structure_group, lambda line: line=='\n'):                 # Group on empty lines
        if not key:
            for i, line in enumerate(sub_group):
                if line.startswith('Relative dose'):
                    data = []   # create a list so can use the append . np arrays must be init with length
                    for line in sub_group:
                         if not line.startswith('Relative dose') and len(line.split()) == 3 :     # 3 cols of data                        
                            aa = [float(i) for i in line.split()]                       
                            #data = np.asarray(data) 
                            data.append(aa)
                
            
                else:
                    info = {};
                    for line in sub_group: 
                        if len(line.split()) == 2 : 
                            field,value=line.split(':')
                            if isinstance(value, str): # if a string
                                value=value.strip()
                            else:
                                value = float(value)
                            info[field]=value
    
    i = i + 1 



from decimal import *
line = '                0                   0                       100'
print(len(line.split()))
bb = [float(i) for i in line.split()]
bb
#results =[float(i) for i in bb]

#results





