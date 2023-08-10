import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

comments = pd.read_csv('Data/comments.csv')
comments = comments['body']

print(comments.shape)

print (comments.head(8))

comments[0].lower() # Convert the first comment to lowercase

comments.str.lower().head(8)  # Convert all comments to lowercase

comments.str.upper().head(8)  # Convert all comments to uppercase

comments.str.len().head(8)

comments.str.split(" ").head(8)

comments.str.strip("[]").head(8)  # Strip leading and trailing bracket

comments.str.cat()[0:500] # Check the first 500 characters

comments.str.slice(0,10).head(8) # Slice the first 10 characters

comments.str[0:10].head(8) # Slice the first 10 characters

comments.str.slice_replace(5, 10, 'Wolves Rule! ').head()

comments.str.replace('Wolves', 'Pups').head(8)

logical_index = comments.str.lower().str.contains('wigg|drew')

comments[logical_index].head(10)  #Get first 10 comments about Wiggins

# calculate the ratio of comments that mention Andrew Wiggins:
len(comments[logical_index])/len(comments)

my_series = pd.Series(['will','bill','Till','still','gull'])

my_series.str.contains('.ill')  # Match any substring ending in ill

my_series.str.contains("[Tt]ill") # Matches T or t followed by "ill"

ex_strl = pd.Series(['Where did he go', 'He went to the mall', 'he is good'])

ex_strl.str.contains('^(He|he)') # Matches He or he at the start of a string

ex_strl.str.contains('(go)$') # Matches go at the end of a string

ex_str2 = pd.Series(["abdominal","b","aa","abbcc","aba"])

# Match 0 or more a's, a single b, then 1 or characters
ex_str2.str.contains('a*b.+')

# Match 1 or more a's, an optional b, then 1 or a's
ex_str2.str.contains('a+b?a+')

ex_str3 = pd.Series(["aabcbcb","abbb","abbaab","aabb"])

ex_str3.str.contains("a{2}b{2,}")   # Match 2 a's then 2 or more b's

ex_str4 = pd.Series(["Mr. Ed","Dr. Mario","Miss\Mrs Granger."])

ex_str4.str.contains(r"\\") #Match strings containing a backslash

comments.str.count(r'[Ww]olve').head(8)

comments.str.findall(r"[Ww]olves").head(8)

web_links = comments.str.contains(r'https?:')

posts_with_links = comments[web_links]

print(len(posts_with_links))

posts_with_links.head(5)

only_links = posts_with_links.str.findall(r"https?:[^ \n\)]+")

only_links.head(10)

