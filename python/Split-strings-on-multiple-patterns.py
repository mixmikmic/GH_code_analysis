'hello you'.split()

'hello | you'.split('|')

# Doesn't seem to like more than one delimiter
'hello | you; and, me'.split(',|;')

import re

# Pipe-separated delimiters
# Both "," and ";" are split on
re.split(',|;', 'hello | you; and, me')

# What if you want to split on "|" too?
# Escape it: \|
re.split(',|;|\|', 'hello | you; and, me')

