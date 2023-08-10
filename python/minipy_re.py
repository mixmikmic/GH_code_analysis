import re

easy_task = "LUMS: 01524 510752"

# slice the string
tel = easy_task[6:]
tel

# \d => 1 digit (0 to 9)
tel = re.search(r'\d\d\d\d\d \d\d\d\d\d\d', easy_task)
print(tel)
print(tel.group(0))

print('hello\nworld')

print(r'hello\nworld')

print('We can use re to extract "%s" from "%s"' % (tel.group(0), easy_task))

print('We can use re to extract "{}" from "{}"'.format(tel.group(0), easy_task))

# my new love!
print(f'We can use re to extract "{tel.group(0)}" from "{easy_task}"')

# again, we can slice the string using the index
print(easy_task[6:12], easy_task[13:])

tel = re.search(r'(\d\d\d\d\d) (\d\d\d\d\d\d)', easy_task)
print(tel.group(1))
print(tel.group(2))

areaCode, mainNumber = tel.groups()
print(areaCode, mainNumber)

hard_task = "LUMS: (0)1524 510752"

# Escape with backslash `\`, such as parenthesis `\(` and `\)`
tel = re.search(r'(\(\d\)\d\d\d\d) (\d\d\d\d\d\d)', hard_task)

areaCode, mainNumber = tel.groups()

print(areaCode, mainNumber)

easy_task = "LUMS: 01524 510752"
hard_task = "LUMS: (0)1524 510752"

# create a pattern using vertial bar indicating alternatives
pattern1 = re.compile(r'(\d\d\d\d\d|\(\d\)\d\d\d\d) (\d\d\d\d\d\d)')

easy_areaCode1, easy_mainNumber1 = pattern1.search(easy_task).groups()
hard_areaCode1, hard_mainNumber1 = pattern1.search(hard_task).groups()

print(easy_areaCode1, hard_areaCode1)

# optional character using ?
pattern2 = re.compile(r'(\(?\d\)?\d\d\d\d) (\d\d\d\d\d\d)')

easy_areaCode2, easy_mainNumber2 = pattern2.search(easy_task).groups()
hard_areaCode2, hard_mainNumber2 = pattern2.search(hard_task).groups()

print(easy_areaCode2, hard_areaCode2)

# match different separater formats
tels = ['01524 510752', '01524-510752', '01524.510752', '01524  510752', '01524510752']

# a dot example
pattern = re.compile(r'\d\d\d\d\d.\d\d\d\d\d\d')

for tel in tels:
    match = pattern.search(tel)
    if match:
        print("Found: {}".format(match.group()))

# match tel number without country code
tels = ['+44 01524 510752', '44 01524-510752', '01524.510752', '01524510752', '(0)1524 510752']

# a carot example
pattern = re.compile(r'^\(?0\)?\d+.?\d+')

for tel in tels:
    match = pattern.search(tel)
    if match:
        print("Found: {}".format(match.group()))

# only match gentlemen (without errors)
names = ['Mr Xi',
         'Mr. Trump', 
         'Mr Trump', 
         'Ms Trump', 
         'Mrs. Trump',
         'Mr rump',
         'Mr. T']

pattern = re.compile(r'Mr\.?\s[A-Z]\w+')

for name in names:
    match = pattern.search(name)
    if match:
        print(match.group())

# only match animals
words = ['hog', 'dog', 'bog']

pattern = re.compile(r'[^b]og')
for word in words:
    match = pattern.search(word)
    if match:
        print(match.group())

# match the last two expressions
words = ['+44 (0)1524 65201',
         '+44 (0)1524 510752',
         '+44 (0)1524 99999999',
         '+44 (0)1524 9999']

pattern = re.compile(r'.*\s\d{5,6}') # problematic

for word in words:
    match = pattern.search(word)
    if match:
        print(match.group())

three_tels = """LUMS general office: +44 (0)1524 510752
Undergraduate enquiries: +44 (0)1524 592938
Postgraduate enquiries: +44 (0)1524 510733"""

pattern = re.compile(r'(\+\d{2})\s(\(?0\)?\d{4})\s(\d{5,6})')

# search returns the first match and ignore all the remainings
match = pattern.search(three_tels)
print(match)

# findall returns a list of matches
matchs = pattern.findall(three_tels)
print(matchs)

top_secret = """Classified, Max clearance level, eyes only: 
Agent Liang pass the extremely secret documents to Special Agent Geogre. 
After 15 sec, this notebook will explodeeee!"""

# let's censor the document log
pattern = re.compile(r'Agent\s\w+')

protected_secret = pattern.sub('YOUKNOWWHO', top_secret)

print(protected_secret)

# Initial only
pattern = re.compile(r'Agent\s(\w)\w*')

censored_secret = pattern.sub(r'\1*****', top_secret)

print(censored_secret)

name1 = 'Firstname Lastname'
name2 = 'Lastname, Firstname'

pattern1 = re.compile(r'([A-Z]\w*)\s([A-Z]\w*)')
swapped_name1 = pattern1.sub(r'\2, \1', name1)
print(swapped_name1)

pattern2 = re.compile(r'([A-Z]\w*),\s([A-Z]\w*)')
swapped_name2 = pattern2.sub(r'\1 \2', name2)
print(swapped_name2)

