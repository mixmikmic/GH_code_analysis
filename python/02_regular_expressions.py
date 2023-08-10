import re

# only exacts match in the given string
re.match(r'\d+',u'1234')

# not match
re.match(r'\d+',u'(16) 3456-4567')

# search will return positive with a given pattern was found in any place inside the given string
re.search(r'\d+',u'(16) 3456-4567')

prog = re.search(r'(\d)+',u'(16) 3456-4567')
if prog:
    print prog.group(0)

# findall willl return all matches with a given pattern inside a given string 
re.findall(r'\d+',u'(16) 3456-4567')

# used in many cases for tokenization
re.findall(r'\w+',u'uma palavra ou outra')

# but it should always be used with Unicode flag!
re.findall(r'\w+',u'maça é uma fruta')

re.findall(r'\w+',u'maça é uma fruta', flags=re.UNICODE+re.IGNORECASE)



