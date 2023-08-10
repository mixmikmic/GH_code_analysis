import re

collection = ['django_migrations.py',
            'django_admin_log.py',
            'main_generator.py',
            'migrations.py',
            'api_user.doc',
            'user_group.doc',
            'accounts.txt',
            ]

def FuzzyFinder(search_term, collection):
    """This function essentially returns the strings that have the specified characters in a sequential order"""
    suggestions = []
    pattern = '.*'.join(search_term)
    compiled_re = re.compile(pattern)
    for item in collection:
        match = compiled_re.search(item)
        if match:
               suggestions.append(item)
    return suggestions
    
FuzzyFinder('mig',collection)

#sorting tuples example
sorted([(5,'bTEST'),(5,'aTEST'),(5,'1TEST'),(7,'TEST'),(3,'TEST'),(1,'TEST')])

def FuzzyFinder(search_term, collection):
    """This function essentially returns the strings in a list that have the specified characters in a sequential order.
    It ranks the strings based on the position of the first occurence of the matching character"""
    suggestions = []
    pattern = '.*'.join(search_term) # Converts 'djm' to 'd.*j.*m'
    compiled_re = re.compile(pattern) # Compiles a regex.
    for item in collection:
        match = compiled_re.search(item) # Checks if the current item matches the regex.
        if match:
               suggestions.append((match.start(),item))
    return [x for _,x in sorted(suggestions)]
    
FuzzyFinder('mig',collection)

collection = ['django_migrations.py',
            'django_admin_log.py',
            'main_generator.py',
            'migrations.py',
            'api_user.doc',
            'user_group.doc',
            'accounts.txt',
            ]

def FuzzyFinder(search_term, collection):
    """This function essentially returns the strings in a list that have the specified characters in a sequential order.
    It ranks the strings based on the compact match lengths and position of the first occurence of the matching character"""
    suggestions = []
    pattern = '.*'.join(search_term) # Converts 'djm' to 'd.*j.*m'
    compiled_re = re.compile(pattern) # Compiles a regex.
    for item in collection:
        match = compiled_re.search(item) # Checks if the current item matches the regex.
        if match:
               suggestions.append((len(match.group()),match.start(), item))
    return [x for _,_,x in sorted(suggestions)]
    
FuzzyFinder('mig',collection)

FuzzyFinder('user', collection)

collection = ['django_migrations.py',
            'django_admin_log.py',
            'main_generator.py',
            'migrations.py',
            'api_user.doc',
            'user_group.doc',
            'accounts.txt',
            ]

def FuzzyFinder(search_term, collection):
    """This function essentially returns the strings in a list that have the specified characters in a sequential order.
    It ranks the strings based on the compact match lengths and position of the first occurence of the matching character"""
    suggestions = []
    pattern = '.*?'.join(search_term) # Converts 'djm' to 'd.*j.*m'
    compiled_re = re.compile(pattern) # Compiles a regex.
    for item in collection:
        match = compiled_re.search(item) # Checks if the current item matches the regex.
        if match:
               suggestions.append((len(match.group()),match.start(), item))
    return [x for _,_,x in sorted(suggestions)]
    
FuzzyFinder('user',collection)

#greedy search
string = "{START} Mary {END} had a {START} little lamb {END} "
pattern = r"{START}.*{END}"
print(re.search(pattern,string).group())
print(re.findall(pattern,string))

#lazy quantifier search
string = "{START} Mary {END} had a {START} little lamb {END} "
pattern = "{START}.*?{END}"
print(re.search(pattern,string).group())
print(re.findall(pattern,string))

# Whitespace stripping
s = '   hello world    \n '
print(s.strip()) #strips whitespace on outside of string
print(s.lstrip()) #strips whitespace on leftside of string
print(s.rstrip()) #strips whitespace on rightside of string

# Character stripping
t = '-----hell-o====='
print(t.lstrip('-')) #stripped specfied characters from the left
print(t.lstrip('=')) #this will do nothing
print(t.strip('-=')) #strip al characters

# lower casing
t = 'HELLO World'
print(t.lower())

#create unicode translation dictionary
str.maketrans('abcdefg','1234567')

# striping non-relevant punctuation
s = 'This is Mikes awesome string'
translation = str.maketrans('aeiou','12345')
s.translate(translation)

#use translation to remove punctuation (replace all punctuation with none... use three args)
s = "This is' a s&tring^ with *alot( of >random !@#punctu*)(@ation)"
translation = str.maketrans("","","!@@#$%>^'&*()")
s.translate(translation)

string = "MikeMikeMikeMikeMike"
print(string.replace("Mike", "Hailey"))
print(string.replace("Mike", "Hailey", 3))

from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

s1 = "New York Mets"
s2 = "New York Meats"

def ratio(s1,s2):
    m = SequenceMatcher(None, s1, s2)
    return(m.ratio())
print(ratio(s1,s2))

#fuzzy match equivalent
print(fuzz.ratio( "New York Mets", "New York Meats"))

print(fuzz.ratio("YANKEES", "NEW YORK YANKEES"))
print(fuzz.ratio("NEW YORK METS", "NEW YORK YANKEES"))

print(fuzz.partial_ratio("YANKEES", "NEW YORK YANKEES"))
print(fuzz.partial_ratio("YANKEEEES", "NEW YORK YANKEES"))
print(fuzz.partial_ratio("NEW YORK METS", "NEW YORK YANKEES"))

#this should produce a sub-perfect match because the shortest string does not perfectly allign to the longer string
a = "NY YANKEES"
b = "NEW YORK YANKEES"
fuzz.partial_ratio(b,a)

##################################### 
##### make string types UNICODE #####
#####################################

def make_type_consistent(s1, s2):
    """If both objects aren't either both string or unicode instances force them to unicode"""
    if isinstance(s1, str) and isinstance(s2, str):
        return s1, s2

    elif isinstance(s1, unicode) and isinstance(s2, unicode):
        return s1, s2

    else:
        return unicode(s1), unicode(s2)

#####################################
####### partial_ratio function ######
#####################################

def partial_ratio(s1, s2):
    """"Return the ratio of the most similar substring
    as a number between 0.0 and 1."""
    s1, s2 = make_type_consistent(s1, s2) #change to unicode!
    
    if len(s1) <= len(s2):
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer=s1
    
    #create a sequence matcher using difflib
    m = SequenceMatcher(None, shorter, longer)
    
    #create match blocks
    blocks = m.get_matching_blocks()
    #print(blocks)
    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3) #shorter starts at 1, longer starts at 3, and block is 3 char long
    #   best score === ratio("abcd", "Xbcd")

    scores = []
    for block in blocks:
        long_start = block[1] - block[0] if (block[1] - block[0])>0 else 0
        long_end = long_start + len(shorter)
        long_substr = longer[long_start:long_end]
        
        m2 = SequenceMatcher(None, shorter, long_substr)
        r = m2.ratio()
        if r > 0.995:
            return 1.0
        else:
            scores.append(r)
    return round(max(scores),3)

print(partial_ratio("YANKEES", "NEW YORK YANKEES"))
print(partial_ratio("NEW YORK METS", "NEW YORK YANKEES"))
print(partial_ratio("METS", "NEW YORK METS"))

print(fuzz.ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets"))
print(fuzz.partial_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets"))

fuzz.token_sort_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets")

import re
import string
def StringProcessor(string_, rejoin=True):
    #clean string and replace non-alphanumeric characters with spaces
    regex = re.compile(r"\W")
    clean_string = regex.sub(" ", string_.strip().lower())
    
    #return tokenized string
    clean_sorted = sorted(clean_string.split())
    
    #join tokens into single string
    if rejoin:
        return " ".join(clean_sorted).strip()
    else:
        return clean_sorted

print(StringProcessor('     BTEsT1*&&&&ATEST2***          '))
print(StringProcessor('New York Mets* vs Atlanta Braves'))
print(StringProcessor('Atlanta Braves vs New York!! Mets!!!'))

def token_sort_ratio(s1, s2, partial=False):
    sorted1 = StringProcessor(s1)
    sorted2 = StringProcessor(s2)
    
    if partial:
        ratio_func = partial_ratio
    else:
        ratio_func = ratio
    
    t = "Mike's fuzzy score: "
    print(t, ratio_func(sorted1, sorted2))
    
token_sort_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets", partial=True)

print("\nFuzzyWuzzy fuzzy score: ",fuzz.token_sort_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets"))

s1 = "mariners vs angels"
s2 = "los angeles angels of anaheim at seattle mariners"

#first try token sort ratio
token_sort_ratio(s1,s2, partial=False)
print("\nFuzzyWuzzy fuzzy score: ",fuzz.token_sort_ratio(s1,s2))

#now try partial token sort ratio
token_sort_ratio(s1,s2, partial=True)
print("\nFuzzyWuzzy fuzzy score: ",fuzz.partial_token_sort_ratio(s1,s2))

def token_set_ratio(s1,s2, partial=False):
    """Find all alphanumeric tokens in each string...
        - treat them as a set
        - construct two strings of the form:
            <sorted_intersection><sorted_remainder>
        - take ratios of those two strings"""
    #create token sets
    tokens1 = set(StringProcessor(s1, rejoin=False))
    tokens2 = set(StringProcessor(s2, rejoin=False))
    
    #parse intersection and differences
    intersection = sorted(tokens1.intersection(tokens2))
    diff1to2 = sorted(tokens1.difference(tokens2))
    diff2to1 = sorted(tokens2.difference(tokens1))
    
    joined_int = " ".join(intersection)
    joined_1to2 = " ".join(diff1to2)
    joined_2to1 = " ".join(diff2to1)
    
    t0 = joined_int.strip()
    t1 = (joined_int + " " + joined_1to2).strip()
    t2 = (joined_int + " " + joined_2to1).strip()
    
    if partial:
        ratio_func = partial_ratio
    else:
        ratio_func = ratio
    
    comps = [ratio_func(t0,t1),
             ratio_func(t0,t2),
             ratio_func(t1,t2)]
    t = "Mike's fuzzy score: "
    print(t, max(comps))

token_set_ratio('this is a new test','this is a different test', partial=False)    

s1 = "mariners vs angels"
s2 = "los angeles angels of anaheim at seattle mariners"

#token_set RATIO
token_set_ratio(s1,s2, partial=False)
print("\nFuzzyWuzzy fuzzy score: ",fuzz.token_set_ratio(s1,s2))

#token_set PARTIAL RATIO
token_set_ratio(s1,s2, partial=True)
print("\nFuzzyWuzzy fuzzy score: ",fuzz.partial_token_set_ratio(s1,s2))

