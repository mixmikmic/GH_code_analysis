name = '2017-03-10-regex'
title = 'Regular expressions and how to use them'
tags = 'basics'
author = 'Maria Zamyatina'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML, Image

html = connect_notebook_to_post(name, title, tags, author)

import re

string = 'Sic Parvis Magna'
pattern = r'.*' # any character as many times as possible

re.search(r'.*', string)

pattern = r'Magna'
re.search(pattern, string)

pattern = r'magna'
re.search(pattern, string)

string = 'Station        : Boulder, CO \n Station Height : 1743 meters \n Latitude       : 39.95'

pattern = r'\d+' # one or more digit
re.search(pattern, string)

re.search(r'\d+\.\d+', string) # float number

re.findall(r'\d+\.\d+|\d+', string) # float or integer number

raw_data = 'O1D = OH + OH : 2.14e-10*H2O;\nOH + O3 = HO2 : 1.70e-12*EXP(-940/TEMP);'
raw_lines = raw_data.split('\n')
raw_lines

m = re.search(r'(.*) (\d)', 'The Witcher 3')
m.group(0)       # entire match

m.group(1)       # first parenthesized subgroup

m.group(2)       # second parenthesized subgroup

m.group(1, 2)    # multiple arguments give us a tuple

for l in raw_lines:
    line = re.search(r'(.*)(.*)', l).group(1, 2)
    print(line)

for l in raw_lines:
    line = re.search(r'(.*)\s:\s(.*);', l).group(1, 2)
    print(line)

alphanum_pattern = r'\w+' # any number or character as many times as possible

for l in raw_lines:
    line = re.search(r'(.*)\s:\s(.*);', l).group(1,2)
    subline_reac, subline_prod = line[0].split('=') # split equation into reactants and products parts using '=' as a separator
    print('Reactants: '+subline_reac, 'Products: '+subline_prod)
    reac = re.findall(alphanum_pattern, subline_reac)
    prod = re.findall(alphanum_pattern, subline_prod)
    print(reac, prod)

eqs = []
for l in raw_lines:
    line = re.search(r'(.*)\s:\s(.*);', l).group(1,2)
    subline_reac, subline_prod = line[0].split('=')
    reac = re.findall(alphanum_pattern, subline_reac)
    prod = re.findall(alphanum_pattern, subline_prod)
    eqs.append(dict(reac=reac, prod=prod, coef=line[1]))
print(eqs)

HTML(html)

