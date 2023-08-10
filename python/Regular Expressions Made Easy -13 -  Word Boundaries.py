import re

string = 'cat catherine catholic  wildcat copycat uncatchable'

pattern = re.compile('cat')

re.findall(pattern, string)

#using space

string = 'cat catherine catholic  wildcat copycat uncatchable'

pattern = re.compile(' cat ')

re.findall(pattern, string)

#only pull out cat with boundary

string = 'cat catherine catholic  wildcat copycat uncatchable'

pattern = re.compile(r'\bcat\b')

re.findall(pattern, string)

#be careful with periods(dot) and non-alphanumeric characters 
#   \w  [A-Za-z0-9_]   \W  +:@^%

string = '.cat catherine catholic  wildcat copycat uncatchable'

pattern = re.compile(r'\bcat\b')

re.findall(pattern, string)

# . = nonalpha numeric

#One side has to have an alphanumeric character and the other side 
#is non alphanumeric character



string = '@cat cat catherine catholic  wildcat copycat uncatchable'

pattern = re.compile(r'\bcat\b')

re.findall(pattern, string)

#Example 2  Twitter examples   Twitter Handles

string = '@moondra2017.org'
string2 = '@moondra'
string3 = 'Python@moondra'
string4 = '@moondra_python'

#we only want @moondra and '@moondra_python' -- string 2 and string 4

pattern = re.compile(r'\b@[\w]+\b')    #no good
re.search(pattern, string)

string = '@moondra2017.org'
string2 = '@moondra'
string3 = 'Python@moondra'
string4 = '@moondra_python'

pattern = re.compile(r'\B@[\w]+\b')    # _  is include in \w
re.search(pattern, string)            # This works but not perfect

string = '@moondra2017.org'
string2 = '@moondra @moondra @moondra'
string3 = 'Python@moondra'
string4 = '@moondra_python'

pattern = re.compile(r'\B@[\w]+\b(?!\.)')
re.findall(pattern, string)

pattern = re.compile(r'\B@[\w]+$')    #  #This is perfect
re.search(pattern, string)

pattern = re.compile(r'\B@[\w]+$') 
re.findall(pattern, string2)

pattern = re.compile(r'\B@[\w]+$') 
re.search(pattern, string3)

pattern = re.compile(r'\B@[\w]+$')
re.search(pattern, string4)





