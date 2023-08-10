import re

import sys

text =  'The quick brown fox jumped over the lazy black bear.'

three_letter_word = '\w{3}'

pattern_re = re.compile(three_letter_word); pattern_re

re_search = re.search('..own', text)

if re_search:
    # Print the search result
    print(re_search.group())

re_match = re.match('..own', text)

if re_match:
    # Print all the matches
    print(re_match.group())
else:
    # Print this
    print('No matches')

re_split = re.split('e', text); re_split

re_sub = re.sub('e', 'E', text, 3); print(re_sub)

