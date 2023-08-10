import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

file_path = '~/Desktop/pcparts/datatables/'

#Function to convert from single size measurements to numbers:
def size_to_num(df, column_str):
    list_val=[]

    for i in range(1, df.shape[0]+1):
        if type(df[column_str][i])==int:
            list_val.append(0)
        else:
            list_val.append(float(re.search('[0-9]{1,2}\.[0-9]+', df[column_str][i]).group(0)))

    return list_val

#Function to grab compatible COOLERS based on CPU socket
def socket_filter(socket):
    return [True if re.search(socket, x) else False for x in cooler['supported sockets']]

#Function to convert from cross-fire/sli words to numeric values
def sli_cross_num(df, column_str):
    list_val=[]
    
    for x in df[column_str]:
        if x == 'No':
            list_val.append(1)
        elif x == 'Yes':
            list_val.append(2)
        else:
            try:
                re.search('(4-way)', x).group(1)
                list_val.append(4)
            except:
                list_val.append(3)
            
    return list_val

#Function to convert range values into [min, max] or [0, only value] list
def limit_finder(df, column_str, ret_type, exp_groups):
    min_list = []
    max_list = []
    
    for el in df[column_str]:
        limit_search = re.search('([0-9]{1,3}[\.]?[0-9])(\s-\s)?([0-9]{1,4}[\.]?[0-9]?)?', el)
        try:
            if limit_search.group(exp_groups) == None:
                min_list.append(ret_type(0))
                max_list.append(ret_type(limit_search.group(1)))
            else:
                min_list.append(ret_type(limit_search.group(1)))
                max_list.append(ret_type(limit_search.group(exp_groups)))
        except:
            min_list.append(ret_type(0))
            max_list.append(ret_type(0))
        
    return [min_list, max_list]

#Take in RAM speed, type and output compatible mobos:
def mobo_ram_compatibility(mem_speed):
    mem_type = int(re.search('^DDR([234])', mem_speed).group(1))
    temp_bool_type = [mem_type == x for x in motherboard['DDR type']]
    
    mem_n = re.search('([0-9]{2,4})', mem_speed).group(1)
    temp_bool_n = [(mem_n in x) for x in motherboard['memory type']]
    return ([a and b for a, b in zip(temp_bool_type, temp_bool_n)])

#Use mobo-psu dictionary to find compatible psu's for given mobo
def mobo_psu_bool(mobo_type):
    psu_comps = mobo_psu_dict[mobo_type]
    for i in range(0, len(psu_comps)):
        bool1 = [bool(re.match(psu_comps[i], x)) for x in psu_type1]
        bool2 = [bool(re.match(psu_comps[i], x)) for x in psu_type2]
        run_bool = [a or b for a, b in zip(bool1, bool2)]
        
        if i > 0:
            run_bool = [a or b for a, b in zip(run_bool, temp_bool)]
        
        temp_bool = run_bool[:]

    return run_bool

mobo_psu_dict = {'Micro ATX':['Micro ATX', 'ATX', 'SFX', 'TFX', 'Flex ATX'],
                 'Mini ITX':['Mini ITX', 'SFX', 'TFX', 'Flex ATX'],
                 'ATX':['ATX', 'EPS'],
                 'EATX':['EPS']}

## Setup for usage ##
# Choice of Motherboard type
mobo_type = 'Mini ITX'
print 'Choice of mobo type:\n{}\n'.format(mobo_type)

# Compatible PSU types
print 'Compatible PSU types:\n{}\n'.format(mobo_psu_dict[mobo_type])

psu_type1 = [re.search('([A-Za-z]{1,5})(12V\s/\s)?([A-Za-z]{1,5})?', x).group(1) for x in psu['type']]

psu_type2 = []
for i in range(1, len(psu['type'])+1):
    try:
        psu_type2.append(str(re.search('([A-Za-z]{1,5})(12V\s/\s)?([A-Za-z]{1,5})?', psu['type'][i]).group(3)))
    except: 
        psu_type2.append('')

case = pd.read_csv('{}case_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
case.fillna(value=0, inplace=True)
case['prod_price'] = case['prod_price'][case['prod_price'] > 0]
case.head()

cpu = pd.read_csv('{}cpu_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
cpu.fillna(value=0, inplace=True)
cpu['prod_price'] = cpu['prod_price'][cpu['prod_price'] > 0]
cpu.head(2)

cooler = pd.read_csv('{}cooler_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
cooler.fillna(value=0, inplace=True)
cooler['prod_price'] = cooler['prod_price'][cooler['prod_price'] > 0]
cooler.head(2)

gpu = pd.read_csv('{}gpu_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
gpu.fillna(value=0, inplace=True)
gpu['prod_price'] = gpu['prod_price'][gpu['prod_price'] > 0]
gpu.head()

memory = pd.read_csv('{}memory_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
memory.fillna(value=0, inplace=True)
memory['prod_price'] = memory['prod_price'][memory['prod_price'] > 0]
memory.head()

motherboard = pd.read_csv('{}motherboard_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
motherboard.fillna(value=0, inplace=True)
motherboard['prod_price'] = motherboard['prod_price'][motherboard['prod_price'] > 0]
motherboard['maximum supported memory'][motherboard['maximum supported memory'] == 0] = '1TB'
motherboard = motherboard[~(motherboard['cpu socket'] == 0)]
motherboard.head()

psu = pd.read_csv('{}psu_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
psu.fillna(value=0, inplace=True)
psu['prod_price'] = psu['prod_price'][psu['prod_price'] > 0]
psu.head()

storage = pd.read_csv('{}storage_dt.csv'.format(file_path), sep=',', engine='python', index_col=0)
storage.fillna(value=0, inplace=True)
storage.head()

case['type'].unique()

case[['rating_n', 'rating_val', 'prod_price']].head(3)

g = sns.PairGrid(case[['rating_n', 'rating_val', 'prod_price', 'type']], hue='type', palette='Set2')
g.map_diag(plt.hist, bins=10)
g.map_offdiag(plt.scatter)
g.add_legend()

list(motherboard['form factor'].unique())

print case['motherboard compatibility'].describe()

mbus = list(motherboard['form factor'].unique())
for mbu in mbus:
    case['{} compatibility'.format(mbu)] = case['motherboard compatibility'].str.contains('{}'.format(mbu))
case.head(5)

#size_to_num defined in first few lines of document
    
case['gpu limit in'] = size_to_num(case, 'maximum video card length')

case.head(1)

#This doesn't require more work, can use simple filters as below
case[case['color'].str.contains('Green')]

#Front panel usb is already well formed
case['front panel usb 3.0 ports'].unique()

#Strip out leading space from column names:
case.columns = [re.sub('^\s','', col_name) for col_name in case.columns]
#Strip out "" symbols from inches columns
case.columns = [re.sub('\"\"', 'in', col_name) for col_name in case.columns]
print case.columns

plt.figure(figsize=(6, 3))
sns.heatmap(case[['prod_price', 'rating_val', 'rating_n',
                         'internal 2.5in bays', 'internal 3.5in bays',
                         'gpu limit in']].corr().round(2), annot=True, annot_kws={"size": 15})



cpu.columns

g = sns.PairGrid(cpu[['rating_n', 'rating_val', 'prod_price', 'cores']], hue='cores', palette='Set2')
g.map_diag(plt.hist, bins=10)
g.map_offdiag(plt.scatter)
g.add_legend()

litho = [int(re.search('[0-9]{1,2}', cpu['lithography'][x]).group(0)) for x in range(1, cpu.shape[0]+1)]

cpu['lithography'] = litho
cpu[['lithography','prod_name','prod_price']].head()

g = sns.PairGrid(cpu[['rating_n', 'rating_val', 'prod_price', 'lithography']], hue='lithography', palette='Set2')
g.map_diag(plt.hist, bins=10)
g.map_offdiag(plt.scatter)
g.add_legend()

sns.jointplot(cpu['rating_val'], cpu['prod_price'])

for el in ['socket', 'l1 cache', 'l2 cache', 'l2 cache-2', 'l3 cache', 'includes cpu cooler']:
    print '{} unique values: {}'.format(el, cpu[el].unique())

###   First work with L1 data   ###
#Grab all of the L1 data out of the L1 column
l1 = [re.search('([0-9]{1,2}) x ([0-9]{1,2})([A-Za-z]{1,2})', cpu['l1 cache'][x]) for x in range(1, cpu.shape[0]+1)]

#Get the multiplier out of the L1 data
l1_mult = [int(l1[x].group(1)) for x in range(0, cpu.shape[0])]
#Likewise for the data-size value
l1_val = [int(l1[x].group(2)) for x in range(0, cpu.shape[0])]
#And finally the data type
l1_type = [str(l1[x].group(3)) for x in range(0, cpu.shape[0])]

for i in range(0, cpu.shape[0]):
    if str(l1[i].group(3)) != 'KB':
        print 'Not L1 in KB'

#All values for L1 are in KB, can multiply both L1 values together without data type problems
cpu['l1 total'] = [x[0]*x[1] for x in zip(l1_mult, l1_val)]
cpu.head(1)

###   Now work with L2 data   ###
#Grab all of the L2 data out of the L2 column
l2 = [re.search('([0-9]{1,2}) x ([0-9]{1,3})([A-Za-z]{1,2})', cpu['l2 cache'][x]) for x in range(1, cpu.shape[0]+1)]

#Get the multiplier out of the L1 data
l2_mult = [int(l2[x].group(1)) for x in range(0, cpu.shape[0])]
#Likewise for the data-size value
l2_val = [int(l2[x].group(2)) for x in range(0, cpu.shape[0])]
#And finally the data type
l2_type = [str(l2[x].group(3)) for x in range(0, cpu.shape[0])]

#Convert MB to KB
l2_val = [int(l2[x].group(2)) if str(l2[x].group(3)) == 'KB' else int(l2[x].group(2))*2**10 for x in range(0, cpu.shape[0])]

#Assign values to dataframe
cpu['l2 total'] = [x[0]*x[1] for x in zip(l2_mult, l2_val)]
cpu.head(1)

###   Next work with L2-2 data   ###
#Grab all of the L2-2 data out of the L2-2 column
l2_2 = [re.search('([0-9]{1,2}) x ([0-9]{1,3})([A-Za-z]{1,2})', cpu['l2 cache-2'][x]) for x in range(1, cpu.shape[0]+1)]

#Get the multiplier out of the L1 data
l2_2_mult = [int(l2_2[x].group(1)) for x in range(0, cpu.shape[0])]
#Likewise for the data-size value
l2_2_val = [int(l2_2[x].group(2)) for x in range(0, cpu.shape[0])]
#And finally the data type
l2_2_type = [str(l2_2[x].group(3)) for x in range(0, cpu.shape[0])]

#Convert MB to KB
l2_2_val = [int(l2_2[x].group(2)) if str(l2_2[x].group(3)) == 'KB' else int(l2_2[x].group(2))*2**10 for x in range(0, cpu.shape[0])]

#Assign values to dataframe
cpu['l2-2 total'] = [x[0]*x[1] for x in zip(l2_2_mult, l2_2_val)]
cpu[['l2 cache-2', 'l2-2 total']].head()

###   Finally work with L2-2 data   ###
#Grab all of the L3 data out of the L3 column
l3 = [re.search('([0-9]{1,2}) x ([0-9]{1,2})([A-Za-z]{1,2})', cpu['l3 cache'][x]) for x in range(1, cpu.shape[0]+1)]

l3_mult = []
l3_val = []
    
for x in range(0, cpu.shape[0]):
    #Get the multiplier out of the L3 data
    try:
        l3_mult.append(int(l3[x].group(1)))
    except:
        l3_mult.append(0)

    #Likewise for the data-size value
    try:
        l3_val.append(int(l3[x].group(2)))
    except: 
        l3_val.append(0)

#All values for L3 are in MB, can multiply both L3 values together without data type problems
cpu['l3 total'] = [x[0]*x[1] for x in zip(l3_mult, l3_val)]
cpu[['l3 cache', 'l3 total']].tail()

#Get all unique values
sockets = list(cpu['socket'].unique())
print sockets

#Most of the entries are fine, need to clean up 'AM3/AM2+'
clean_socket = []
for el in cpu['socket']:
    try:
        re.search('(AM3/AM2\+)', el).group(0)
        clean_socket.append('AM2+-AM3')
    except:
        clean_socket.append(el)
cpu['socket'] = clean_socket

cpu['tdp'] = [re.search('([0-9]{2,3})', x).group(1) for x in cpu['thermal design power']]

cpu['op freq n'] = [float(re.search('([0-9]\.[0-9])', x).group(1)) for x in cpu['operating frequency']]

max_turbo_list = []
for x in cpu['max turbo frequency']:
    try:
        max_turbo_list.append(float(re.search('([0-9]\.[0-9])', x).group(1)))
    except:
        max_turbo_list.append(float(0))
        
cpu['turbo freq n'] = max_turbo_list

cpu[['op freq n', 'turbo freq n']][1:5]

cpu_n = cpu[cpu['prod_price'] < np.mean(cpu['prod_price']) + 2*np.std(cpu['prod_price'])]

plt.figure(figsize=(6, 3))
sns.heatmap(cpu_n[['prod_price', 'rating_val',
                 'l1 total', 'l2 total', 'l2-2 total', 'l3 total'
                ]].corr().round(2), annot=True, annot_kws={"size": 15})

g = sns.PairGrid(cpu_n[['prod_price', 'l1 total', 'l3 total', 'lithography']], hue='lithography', palette='Set2')
g.map_diag(plt.hist, bins=10)
g.map_offdiag(plt.scatter)
g.add_legend()

plt.figure(figsize=(6, 3))
sns.heatmap(cpu_n[['prod_price', 'rating_val',
                 'op freq n', 'turbo freq n',
                 'lithography'
                ]].corr().round(2), annot=True, annot_kws={"size": 15})



cooler.columns

cooler_filter = cooler[['rating_n', 'rating_val', 'prod_price', 'radiator size']][cooler['radiator size'] != ' ']
g = sns.PairGrid(cooler_filter, hue='radiator size', palette='Set2')
g.map_diag(plt.hist, bins=10, normed=False)
g.map_offdiag(plt.scatter)
g.add_legend()

#Take a look at the height column first
cooler_limit=[]
for i in range(1, cooler.shape[0]+1):
    if type(cooler['height'][i])==int:
        cooler_limit.append(0.0)
    else:
        cooler_limit.append(float(re.search('[0-9]{1,2}\.[0-9]+', cooler['height'][i]).group(0)))

cooler['numeric height'] = cooler_limit

#Apply same correction to radiator size column
cooler_limit=[]
for i in range(1, cooler.shape[0]+1):
    if cooler['radiator size'][i]==' ':
        cooler_limit.append(0)
    else:
        cooler_limit.append(int(re.search('[0-9]{1,3}', cooler['radiator size'][i]).group(0)))

cooler['numeric size'] = cooler_limit
print cooler['numeric size'].unique()

print cooler['liquid cooled'].head()
cooler['liquid cooled'] = [True if x == ' Yes' else False for x in cooler['liquid cooled']]
print cooler['liquid cooled'].head()

print cooler['noise level'].head()
print '\n'

#Limit finder return a list of lists: first list is min
cooler['noise level min'] = limit_finder(cooler,'noise level', float, 3)[0]
#Second list is max noise level (or nominal if min is missing)
cooler['noise level max'] = limit_finder(cooler,'noise level', float, 3)[1]

print cooler[['noise level', 'noise level min', 'noise level max']].head()

#Has same format as noise level information
print cooler['fan rpm'].head()
print '\n'

#Limit finder return a list of lists: first list is min
cooler['fan rpm min'] = limit_finder(cooler,'fan rpm', int, 3)[0]
#Second list is max noise level (or nominal if min is missing)
cooler['fan rpm max'] = limit_finder(cooler,'fan rpm', int, 3)[1]

print cooler[['fan rpm', 'fan rpm min', 'fan rpm max']].head()

sockets = list(cpu['socket'].unique())
print 'unique sockets from cpu list:\n {} \n'.format(sockets)

def cooler_tests(df, column_str, specific_arg, general_arg):
    if len(df[column_str][~(df[column_str].str.contains(specific_arg)) &
                                       (df[column_str].str.contains(general_arg))]) == 0:
        return 'If {} is supported, {} is also supported \n'.format(general_arg, specific_arg)
    
    else:
        return 'Need to separate {} & {}\n'.format(general_arg, specific_arg)

for el in [['LGA2011-3', 'LGA2011'],
           ['FM2+', 'FM2'],
           ['AM3+', 'AM2'],
           ['AM4', 'AM3']]:
    print cooler_tests(cooler, 'supported sockets', el[0], el[1])
#Lets generalize this to figure out what our minimum number of values is

def cooler_tests_v2(df, column_str, df2, column_str2, general_arg):
    return_list = []
    for el in df2[column_str2].unique():
        if len(df[column_str][~(df[column_str].str.contains(el)) &
                                           (df[column_str].str.contains(general_arg))]) == 0:
            return_list.append('If {} is supported, {} is also supported \n'.format(general_arg, el))
            if len(df[column_str][~(df[column_str].str.contains(general_arg)) &
                                               (df[column_str].str.contains(el))]) == 0:
                return_list.append('If {} is supported, {} is also supported \n'.format(el, general_arg))
            else: return_list.append('Need to separate {} & {}\n'.format(el, general_arg))
        else:
            return_list.append('Need to separate {} & {}\n'.format(general_arg, el))
            if len(df[column_str][~(df[column_str].str.contains(general_arg)) &
                                               (df[column_str].str.contains(el))]) == 0:
                return_list.append('If {} is supported, {} is also supported \n'.format(el, general_arg))
            else:
                return_list.append('Need to separate {} & {}\n'.format(el, general_arg))
    return return_list

#for el in list(cpu['socket'].unique()):
    #for x in cooler_tests_v2(cooler, 'supported sockets', cpu, 'socket', 'AM1'):
        #print x
        
#AM1, AM4, LGA775, LGA1366 have no equal
#LGA1151, LGA1150, LGA1155, LGA1156 are interchangeable
#AM3, AM3+, FM2+, FM2 are interchangeable
#LGA2011, LGA2011-3 are interchangeable
#If AM2+-AM3 is supported, everything is supported

unique_sockets = ['AM1', 'AM4', 'LGA775', 'LGA1366', 'AM2', 'AM3', 'LGA1151', 'LGA2011']
print unique_sockets

def socket_filter(socket):
    return [True if re.search(socket, x) else False for x in cooler['supported sockets']]

print '{} socket for chosen CPU'.format(cpu['socket'][203])
print '\n'
print 'compatible sockets for chosen cooler: \n{}'.format(cooler['supported sockets'][socket_filter(cpu['socket'][203])][1])

cooler.head(1)



cooler_n = cooler[cooler['prod_price'] < np.mean(cooler['prod_price']) + 2*np.std(cooler['prod_price'])]
cooler_n = cooler[cooler['prod_price'] > 0]

plt.figure(figsize=(6, 3))
sns.heatmap(cooler_n[['prod_price', 'rating_val',
                      'fan rpm min', 'fan rpm max',
                      'noise level min', 'noise level max',
                      'numeric size'
                ]].corr().round(2), annot=True, annot_kws={"size": 13})



print gpu.columns

gpu_filter = gpu[gpu['memory size'].isin(['12GB', '11GB', '8GB', '6GB'])]
g = sns.PairGrid(gpu_filter[['rating_n', 'rating_val', 'prod_price', 'memory size']], hue='memory size', palette='Set2')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

gpu_filter = gpu[gpu['rating_n'] > 3]
sns.jointplot(gpu_filter['rating_val'], gpu_filter['prod_price'])

gpu_filter = gpu[gpu['rating_n'] > 3]
g = sns.PairGrid(gpu_filter[['rating_val', 'prod_price', 'sli support', 'crossfire support']], hue='sli support', palette='Set2')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

for el in ['tdp', 'memory type', 'sli support', 'crossfire support', 'supports g-sync']:
    print '{} unique values: {}'.format(el, gpu[el].unique())

#Need to rescrape html for 'supports freesync' if we want to use g-sync info

#TDP is related to energy used under load
tdp_vals = [int(re.search('([0-9]{2,3})', gpu['tdp'][x]).group(1)) for x in range(1, gpu.shape[0]+1)]
gpu['tdp'] = tdp_vals

#Length is in inches, need to strip out unnecessary parts
gpu['length'] = size_to_num(gpu, 'length')

#Convert from text to number of gpus allowed (sli or crossfire) and store the max of the two numbers
gpu['max gpus'] = map(lambda x: max([x[0],x[1]]),
                      zip(sli_cross_num(gpu, 'sli support'),
                          sli_cross_num(gpu, 'crossfire support')))

gpu['interface'].unique()

gpu['memory size'].unique()

print 'Unique values of "memory size" column:\n{}\n'.format(gpu['memory size'].unique())

#Look for GB and MB identifier, store the number value in the first group; store the unit in the second/third group
gpu_mem = [re.search('([0-9]+)(MB)?(GB)?', x) for x in gpu['memory size']]

#Multiply by the appropriate conversion
gpu['mem int'] = [float(x.group(1))/(2**10) if x.group(2)=='MB' else int(x.group(1)) for x in gpu_mem]

print 'Unique values of "mem int" column:\n{}\n'.format(gpu['mem int'].unique())

gpu['core clock'].unique()

#Look for GHz and MHz identifier, store the number value in the first group; store the unit in the second/third group
gpu_cc = [re.search('([0-9]+)(M)?(G)?Hz', x) for x in gpu['core clock']]

#Multiply by the appropriate conversion
gpu['core n'] = [float(x.group(1))/(2**10) if x.group(2)=='M' else 1+float(x.group(1))/100 for x in gpu_cc]

print 'First few values of "core n" column:\n{}\n'.format(gpu[['core clock', 'core n']][0:5])

gpu['boost clock'].unique()

#Look for GHz and MHz identifier, store the number value in the first group; store the unit in the second/third group
gpu_bc = []

for boost in gpu['boost clock']:
    try:
        gpu_bc.append(1+float(re.search('([0-9]+)(M)?(G)?Hz', boost).group(1))/100)
    except:
        gpu_bc.append(float(0))

#Multiply by the appropriate conversion
gpu['boost n'] = gpu_bc

print 'First few values of "boost n" column:\n{}\n'.format(gpu[['boost n', 'boost clock']][0:5])

gpu[['length','tdp', 'max gpus', 'mem int', 'core n', 'boost n']].head()

plt.figure(figsize=(6, 3))
sns.heatmap(gpu[['prod_price', 'rating_val',
                      'mem int', 'max gpus',
                      'core n', 'boost n',
                      'tdp', 'length'
                ]].corr().round(2), annot=True, annot_kws={"size": 13})

sns.jointplot(gpu['prod_price'], gpu['tdp'], size=5)

print memory.columns

memory_filter = memory[memory['rating_n']>1]
g = sns.jointplot(memory_filter['rating_val'], memory_filter['prod_price'])

memory['price/gb dollars'] = [float(re.search('\$([0-9]{1,4}\.[0-9]+)', x).group(1)) for x in memory['price/gb']]
print memory[['price/gb dollars', 'price/gb']].tail(2)

memory_filter = memory[memory['rating_n']>1]
g = sns.jointplot(memory_filter['rating_val'], memory_filter['price/gb dollars'])

#Speed is already good for motherboard compatibility check
memory['speed'].unique()

#For SQL use later, we need to convert to type and speed
memory['speed type'] = [int(re.search('DDR([0-9])', x).group(1)) for x in memory['speed']]
memory['speed n'] = [int(re.search('-([0-9]{3,4})', x).group(1)) for x in memory['speed']]
memory[['speed', 'speed type', 'speed n']][0:4]

#Find unique values of memory type
print memory['type'].unique()

#Create new column with numeric values of the type column for easy filtering
memory['mem type'] = [int(re.search('([0-9]{3,})', x).group(1)) for x in memory['type']]
print memory[['type', 'mem type']].head(3)

#Filter out columns that are not 240 or 288 pin
memory = memory[memory['mem type'].isin([240, 288])]
print memory['type'].unique()

memory['size'].unique()

### From above output, see that we need to grab the number before GB and the first number in parentheses ###

#First grab the leading number
memory['total mem'] = [int(re.search('^([0-9]+)', x).group(1)) for x in memory['size']]

#Then grab the multiplier
memory['num sticks'] = [int(re.search('\(([0-9])', x).group(1)) for x in memory['size']]

memory[['size', 'total mem', 'num sticks']].head()

plt.figure(figsize=(6, 3))
sns.heatmap(memory[['prod_price', 'rating_val',
                 'total mem', 'num sticks',
                 'speed n', 'price/gb dollars'
                ]].corr().round(2), annot=True, annot_kws={"size": 13})

sns.jointplot(memory['prod_price'], memory['total mem'], size=3.5)

memory_filter = memory[memory['total mem'] < 60]
sns.jointplot(memory_filter['prod_price'], memory_filter['total mem'], size=3.5)



print motherboard.columns

#Found out NaN price value that slipped through earlier filtering
motherboard = motherboard[~motherboard['prod_price'][[x != 0 for x in motherboard['prod_price']]].isnull()]

mobo_filtern = motherboard[motherboard['rating_n'] > 3]
sns.lmplot("rating_val", "prod_price", mobo_filtern, hue="sli support",
           palette={"Yes": "b", "No": "r", "3-way SLI": "g", "4-way SLI": "y"})

mobo_filtern = motherboard[motherboard['rating_n'] > 3]
sns.lmplot("rating_val", "prod_price", mobo_filtern, hue="crossfire support", palette={"Yes": "b", "No": "r"})

#Print out some columns of interest:
print motherboard[['memory type', 'form factor',
                   'maximum supported memory', 'sli support',
                   'crossfire support', 'cpu socket',
                   'raid support', 'onboard ethernet',
                   'memory slots']].head()

### We cleaned the case dataframe to make this easier!
print 'Unique values of "form factor" column from mobo:\n{}\n'.format(motherboard['form factor'].unique())
print 'Boolean columns from case dataframe:\n{}'.format(list(case.columns)[-6:-1])



#Example filtering usage:
print '\nType of mother board:\n{}\n'.format(motherboard['form factor'][200])
print 'Subset of cases that are compatible:\n{}'.format(case[['prod_name', 'motherboard compatibility']][case['{} compatibility'.format(motherboard['form factor'][200])]].head())

print 'Unique values of "maximum supported memory" column:\n{}\n'.format(motherboard['maximum supported memory'].unique())

#Look for GB and TB identifier, store the number value in the first group; store the unit in the second/third group
mobo_mem = [re.search('([0-9]+)(GB)?(TB)?', x) for x in motherboard['maximum supported memory']]

#Multiply by the appropriate conversion
motherboard['max mem int'] = [int(x.group(1))*2**10 if x.group(3)=='TB' else int(x.group(1)) for x in mobo_mem]

print 'Unique values of "max mem int" (converted) column:\n{}\n'.format(motherboard['max mem int'].unique())

#The slashed values for this column actually indicate a range. The max value is all that's needed in practice
## The number of ports is also useful
print 'Unique values of "onboard ethernet" column:\n{}\n\n'.format(motherboard['onboard ethernet'].unique())

#Grab all of the values in the column
ethernet_list = [re.search('([0-9]) x ([0-9]*/)+([0-9]*) Mbps', x) for x in motherboard['onboard ethernet']]

#Example 1: standard 3 element slash
#Show all of the elements
print ethernet_list[1].group(0)
#Show the elements of interest
print [ethernet_list[1].group(1), ethernet_list[1].group(3)]
print '\n'

#Example 2: short onboard ethernet list
#Show all of the elements
print ethernet_list[82].group(0)
#Show the elements of interest
print [ethernet_list[82].group(1), ethernet_list[82].group(3)]

#Assign these values to the dataframe:
### First, get the number of ports
motherboard['ethernet port n'] = [int(x.group(1)) for x in ethernet_list]

print 'Unique values for number of ports:\n{}\n'.format(motherboard['ethernet port n'].unique())

### Next, assign the max value for ethernet speed
motherboard['ethernet port lim'] = [int(x.group(3)) for x in ethernet_list]

print 'Unique values for speed limit, ethernet port:\n{}\n'.format(motherboard['ethernet port lim'].unique())

print 'Unique values of "cpu socket" column:\n{}\n'.format(motherboard['cpu socket'].unique())

#Search for required CPU-MOBO compatibility elements
mobo_cpu = [re.search('([0-9]+)?( x )?([A-Z]{1,3}[0-9]{1,4}\+?-?3?)', x) for x in motherboard['cpu socket']]

#Grab the number of supported CPUs per mobo
motherboard['cpu socket n'] = [1 if x.group(1) == None else int(x.group(1)) for x in mobo_cpu]
print 'Unique values of "cpu socket n" column:\n{}\n'.format(motherboard['cpu socket n'].unique())

#Grab the type of supported CPUs for each mobo
motherboard['cpu socket type'] = [x.group(3) for x in mobo_cpu]
print 'Unique values of "cpu socket type" column:\n{}\n'.format(motherboard['cpu socket type'].unique())

#For reference, here's the available socket types from the cpu dataframe
### These are strict compatibility requirements:
##### 1. must exactly match LGAs
##### 2. an AM or FM mobo with a + can handle both + and normal types
####### e.g. AM2+ mobo can handle AM2 or AM2+ cpu
print cpu['socket'].unique()

def mobo_cpu_socket_bool(socket):
    if socket == 'AM2+-AM3':
        return motherboard['cpu socket type'] == 'AM3'
    else:
        x = re.search('([A-Z]{1,3}[0-9]{1,4}\+?-?3?)', socket)
        return motherboard['cpu socket type'] == x.group(0)

motherboard[mobo_cpu_socket_bool('AM2+-AM3')]

# There is no overlap between DDR3 and DDR4 compatibility
print '5 values from "memory type" column:\n{}\n'.format(motherboard['memory type'][10:16])

### Getting the type is very easy (search for DDR# where # is what we want)
motherboard['DDR type'] = [int(re.search('^DDR([234])', x).group(1)) for x in motherboard['memory type']]
# We're half way done:
motherboard[['DDR type', 'memory type']].head(2)

# Now work on extracting the speed compatibility
# Here is what the input will be
print memory['speed'][1:10]

#Take in RAM speed, type and output compatible mobos:
def mobo_ram_compatibility(mem_speed):
    mem_type = int(re.search('^DDR([234])', mem_speed).group(1))
    temp_bool_type = [mem_type == x for x in motherboard['DDR type']]
    
    mem_n = re.search('([0-9]{2,4})', mem_speed).group(1)
    temp_bool_n = [(mem_n in x) for x in motherboard['memory type']]
    return ([a and b for a, b in zip(temp_bool_type, temp_bool_n)])

#Example usage with specific speed (from memory['speed'])
motherboard[mobo_ram_compatibility('DDR4-1600')].head(3)

print 'Type of first four memory sticks\n{}'.format(memory['speed'][0:4])
print '\n'
print 'Number of compatible mobos for first four memory sticks:\n{}'.format([len(motherboard[mobo_ram_compatibility(x)]) for x in memory['speed']][0:4])

print 'Choices for sli support:\n{}\n'.format(motherboard['sli support'].unique())
print 'Choices for crossfire support:\n{}'.format(motherboard['crossfire support'].unique())

#Convert from text to number of gpus allowed (sli or crossfire) and store the max of the two numbers
motherboard['max gpus'] = map(lambda x: max([x[0],x[1]]),
                              zip(sli_cross_num(motherboard, 'sli support'),
                                  sli_cross_num(motherboard, 'crossfire support')))

motherboard[['max gpus', 'crossfire support', 'sli support']][21:26]

sns.heatmap(motherboard[['rating_val', 'rating_n',
                         'DDR type', 'cpu socket n',
                         'prod_price', 'max mem int',
                         'memory slots', 'ethernet port n',
                         'sata 6 gb/s', 'sata express',
                         'onboard usb 3.0 header(s)', 'max gpus', 'u.2']].corr().round(2), annot=True)

plt.figure(figsize=(6, 3))
sns.heatmap(motherboard[['prod_price','rating_val',
                         'max mem int',
                         'sata 6 gb/s', 'sata express',
                         'onboard usb 3.0 header(s)', 'u.2', 'max gpus'
                ]].corr().round(2), annot=True, annot_kws={"size": 13})

psu.head(2)

print psu.columns

psu_filtern = psu[psu['rating_n'] > 3]
sns.lmplot("rating_val", "prod_price", psu_filtern)

psu['wattage'].unique()

psu['watt n'] = [int(re.search('[0-9]{1,4}', x).group(0)) for x in psu['wattage']]

psu[['watt n', 'wattage']].head(3)

##### Looking at motherboard-PSU compatibility #####

#Format of relations is PSU: motherboard compatibility
# ATX: ATX, microATX
# SFX: microATX, miniITX
# EPS: ATX, EATX
# TFX: microATX, MiniITX
# Flex ATX: microATX, MiniITX
# Micro and mini entries mapped to themselves only
### Convert the above to a dictionary ###

psu_mobo_dict = {'Micro ATX':['Micro ATX'],
                 'Mini ITX':['Mini ITX'],
                 'ATX':['ATX', 'Micro ATX'], 
                 'SFX':['Micro ATX', 'Mini ITX'], 
                 'EPS':['ATX', 'EATX'],
                 'TFX':['Micro ATX', 'Mini ITX'],
                 'Flex ATX':['Micro ATX', 'Mini ITX']}

mobo_psu_dict = {'Micro ATX':['Micro ATX', 'ATX', 'SFX', 'TFX', 'Flex ATX'],
                 'Mini ITX':['Mini ITX', 'SFX', 'TFX', 'Flex ATX'],
                 'ATX':['ATX', 'EPS'],
                 'EATX':['EPS']}

## Example usage ##
# Choice of Motherboard type
mobo_type = 'Mini ITX'
print 'Choice of mobo type:\n{}\n'.format(mobo_type)

# Compatible PSU types
print 'Compatible PSU types:\n{}\n'.format(mobo_psu_dict[mobo_type])

psu_type1 = [re.search('([A-Za-z]{1,5})(12V\s/\s)?([A-Za-z]{1,5})?', x).group(1) for x in psu['type']]

psu_type2 = []
for i in range(1, len(psu['type'])+1):
    try:
        psu_type2.append(str(re.search('([A-Za-z]{1,5})(12V\s/\s)?([A-Za-z]{1,5})?', psu['type'][i]).group(3)))
    except: 
        psu_type2.append('')

# Search for all compatible mobos
def mobo_psu_bool(mobo_type):
    psu_comps = mobo_psu_dict[mobo_type]
    for i in range(0, len(psu_comps)):
        bool1 = [bool(re.match(psu_comps[i], x)) for x in psu_type1]
        bool2 = [bool(re.match(psu_comps[i], x)) for x in psu_type2]
        run_bool = [a or b for a, b in zip(bool1, bool2)]
        
        if i > 0:
            run_bool = [a or b for a, b in zip(run_bool, temp_bool)]
        
        temp_bool = run_bool[:]

    return run_bool

print 'A few compatible PSU\'s for this Mobo:'
psu[mobo_psu_bool(mobo_type)].tail()

#Assign compatibility to dataframe
for mobo in mobo_psu_dict.keys():
    psu['{} compatibility'.format(mobo)] = mobo_psu_bool(mobo)
    print psu['{} compatibility'.format(mobo)].head(1)

power_rating = []

for efficiency in psu['efficiency certification']:
    try:
        output = re.search('80\+\s?([A-Za-z]+)?', efficiency).group(1)
        if output == None:
            power_rating.append('Basic')
        else:
            power_rating.append(output)
    except:
        power_rating.append('Unknown')
    
psu['rating str'] = power_rating
psu[['efficiency certification', 'rating str']][3:7]
psu['rating str'].unique()

plt.figure(figsize=(6, 3))
sns.heatmap(psu[['prod_price','rating_val',
                 'watt n'
                ]].corr().round(2), annot=True, annot_kws={"size": 20})

g = sns.PairGrid(psu[['rating_val', 'prod_price', 'watt n', 'rating str']], hue='rating str', palette='Set2', size = 3.5)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

storage.columns

for el in ['capacity', 'interface', 'form factor', 'cache', 'nand flash type']:
    print 'Unique values of {}:\n{}\n'.format(el, storage[el].unique())

storage_filter = storage[storage['rating_n'] > 1]
g = sns.PairGrid(storage_filter[['rating_n', 'rating_val', 'prod_price', 'form factor']], hue='form factor', palette='Set2')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# Grab all of the data out of the capacity column
capacity = [re.search('([0-9]{1,3}\.?[0-9]?)([A-Za-z]{1,2})', storage['capacity'][x]) for x in range(1, storage.shape[0]+1)]

#Grab the value first
capacity_val = [float(capacity[x].group(1)) for x in range(0, storage.shape[0])]
#And then the data type
capacity_type = [str(capacity[x].group(2)) for x in range(0, storage.shape[0])]

#Convert MB to KB
capacity_val = [float(capacity[x].group(1)) if str(capacity[x].group(2)) == 'GB' else float(capacity[x].group(1))*2**10 for x in range(0, storage.shape[0])]

#Assign values to dataframe
storage['capacity total'] = capacity_val
storage.sort_values(by='capacity total', ascending=False).head(2)

storage['prod_price'] = storage['prod_price'][storage['prod_price'] > 0]
storage = storage[storage['price/gb']!=0]

storage['interface'].unique()

#Assign boolean column for most frequently used interface: SATA
storage['SATA compatible'] = [bool(re.search('SATA', x)) for x in storage['interface']]

#Strip out the parenth'd part of 'U.2 (SFF-8639)' and conver U to lower
storage['interface'] = [re.sub('(\s\(SFF-8639\))','', x) for x in storage['interface']]
storage['interface'] = [re.sub('U.2','u.2', x) for x in storage['interface']]

#Likewise for the M.2 entries
storage['interface'] = [re.sub('(\s\(SFF-8639\))','', x) for x in storage['interface']]
storage['interface'] = [re.sub('U.2','u.2', x) for x in storage['interface']]

storage[['interface']].head()

motherboard[['sata 6 gb/s','sata express', 'u.2']].sort_values(ascending=False, by=['u.2', 'sata express']).head()

storage['2.5in form'] = [bool(re.search('2\.5', x)) for x in storage['form factor']]
storage['3.5in form'] = [bool(re.search('3\.5', x)) for x in storage['form factor']]

storage[storage['2.5in form']].head(1)

#Dollar sign is always first character, slice it off with list comprehension
storage['price/gb'] = [x[1:] for x in storage['price/gb']]



plt.figure(figsize=(6, 3))
sns.heatmap(storage[['prod_price','rating_val',
                 'capacity total'
                ]].corr().round(2), annot=True, annot_kws={"size": 20})

storage_filter = storage[storage['3.5in form'] == True]
sns.jointplot("prod_price", "capacity total", storage_filter)

storage_filter = storage[storage['2.5in form'] == True]
sns.jointplot("prod_price", "capacity total", storage_filter)

from sqlalchemy import create_engine
#Create engine, write with to_sql pandas dataframe method for each dataframe
db_engine = create_engine('sqlite:///pcpart.db')

for el in ['case', 'cpu', 'cooler', 'gpu', 'memory', 'motherboard', 'psu', 'storage']:
    eval(el).to_sql('{}_table'.format(el), db_engine, if_exists='replace')

from sqlalchemy import create_engine
db_engine = create_engine('sqlite:///pcpart.db')
motherboard.to_sql('motherboard_table', db_engine, if_exists='replace')

import sqlite3

con = sqlite3.connect('pcpart.db')
cursor = con.cursor()

#Code to print out all available tables in our database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print cursor.fetchall()

#Example read from Storage table in database
df = pd.read_sql_query("SELECT prod_name as name,"
                       "prod_price as price,"
                       "`capacity total` as capacity "
                       "FROM storage_table "
                       "WHERE `2.5in form` = 1 "
                       "AND `price/gb` < 0.5 "
                       "ORDER BY `price/gb`", con)
df.head()



