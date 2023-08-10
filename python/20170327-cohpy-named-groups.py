m.group(2)

m.group('first_name')

import re

foo_pattern = re.compile('''
    ^
    ([A-Za-z]+)
    ,[ ]
    ([A-Za-z]+)
    $
''', re.VERBOSE)

s = 'James, Mackenzie'

m = re.match(foo_pattern, s)
m

m.groups

m.group(0)

m.group(1)

m.group(2)

foo_pattern = re.compile('''
    ^
    (?P<last_name>[A-Za-z]+)
    ,[ ]
    (?P<first_name>[A-Za-z]+)
    $
''', re.VERBOSE)

m = re.match(foo_pattern, s)
m

m.groups

m.group(0)

m.group(1)

m.group(2)

m.group('last_name')

m.group('first_name')

