# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi
import re

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)

phone_re = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
text  = "Call me at 382-384-3840."
match = re.search(phone_re, text)
match

if re.search(phone_re, text):
    print("Found a match!")

if re.search(phone_re, 'Hello world'):
    print("No match; this won't print")

gmail_re = r'[a-zA-Z0-9]+@gmail\.com'
text = '''
From: email1@gmail.com
To: email2@yahoo.com and email3@gmail.com
'''
re.findall(gmail_re, text)

phone_re = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
text  = "Sam's number is 382-384-3840 and Mary's is 123-456-7890."
re.findall(phone_re, text)

# Same regex with parentheses around the digit groups
phone_re = r"([0-9]{3})-([0-9]{3})-([0-9]{4})"
text  = "Sam's number is 382-384-3840 and Mary's is 123-456-7890."
re.findall(phone_re, text)

messy_dates = '03/12/2018, 03.13.18, 03/14/2018, 03:15:2018'
regex = r'[/.:]'
re.sub(regex, '-', messy_dates)

toc = '''
PLAYING PILGRIMS============3
A MERRY CHRISTMAS===========13
THE LAURENCE BOY============31
BURDENS=====================55
BEING NEIGHBORLY============76
'''.strip()

# First, split into individual lines
lines = re.split('\n', toc)
lines

# Then, split into chapter title and page number
split_re = r'=+' # Matches any sequence of = characters
[re.split(split_re, line) for line in lines]

# HIDDEN
text = '''
"Christmas won't be Christmas without any presents," grumbled Jo, lying on the rug.
"It's so dreadful to be poor!" sighed Meg, looking down at her old dress.
"I don't think it's fair for some girls to have plenty of pretty things, and other girls nothing at all," added little Amy, with an injured sniff.
"We've got Father and Mother, and each other," said Beth contentedly from her corner.
The four young faces on which the firelight shone brightened at the cheerful words, but darkened again as Jo said sadly, "We haven't got Father, and shall not have him for a long time."
'''.strip()
little = pd.DataFrame({
    'sentences': text.split('\n')
})

little

quote_re = r'"[^"]+"'
little['sentences'].str.findall(quote_re)

# Extract text within double quotes
quote_re = r'"([^"]+)"'
spoken = little['sentences'].str.extract(quote_re)
spoken

little['dialog'] = spoken
little

print(little.loc[4, 'sentences'])

print(little.loc[4, 'dialog'])

