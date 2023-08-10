text= "This is a string literal that has a quote \" character." 
print(text)

text= "This has a quote \" char followed by \n\n\n three new lines!!." 
print(text)

raw_text= r"This has a quote \" char followed by \n\n\n three new lines!!." 
print(raw_text)

# match() will only find matches if they occur at the beginning of 
# the searched string:
import re
text="apple berry orange berry"
re.match(r'apple',text)

#The above means there is a match and Python is returning the matching 
# Object. 
# We can access the matched pattern with: group(0)
my_match=re.match(r'apple',text)
my_match.group(0)

# Since "berry" is not in the beginning of the string, there will be
# no match.
print(re.match(r'berry',text))

# search() is like match(), excpet that it is not restricted to finding a match
# at the beginning: It will find a match anywhere in the string:
print(re.search(r'berry',text))
print(re.search(r'apple',text))
print(re.search(r'orange',text))

# Note that search() stops looking after it finds the first match.
# As such, even though there are wto examples of the string "berry",
# match() only returns one match (the first match)
my_berry_match=re.search(r'berry',text)
my_berry_match.group(0)

# We can actually access the indexes of the matched "berry" string:
start=my_berry_match.start()
end=my_berry_match.end()
print("Start index: %s" % start)
print("End   index: %s" % end)

print(text[6:11])

# findall() is like search(), but is exhaustive: It finds all the matches
all_berry_matches=re.findall(r'berry',text)
print(all_berry_matches)

# Since it returns a list of what matched, findall() does not work with
# grouping. Instead, just access each item in the returned list as 
# what would have been a group 
all_berry_matches[0]

# We can surround certain surround certain parts of the regex in paranthese
# and access them later on via group numbers
tweet="This is a tweet with #hashtag1 and #hashtag2 https://cnn.com"
my_hashtags=re.search(r'(#\S+)\s+\S+\s+(#\S+)', tweet)
print(my_hashtags.group(1)) # whatever is in the first ()
print(my_hashtags.group(2)) # whatever is in the second ()

print(my_hashtags)

# We can surround certain surround certain parts of the regex in paranthese
# and access them later on via group numbers
tweet="This is a tweet with #hashtag1 and #hashtag2 https://cnn.com"
my_hashtags=re.search(r'(#\S+)(?P<my_and_group>\s+\S+\s+)(#\S+)', tweet)
print(my_hashtags.group("my_and_group")) 

#tweet="This is a tweet with #hashtag1 and #hashtag2 https://cnn.com"
tweet_modified="This is a tweet with #hashtag1 #hashtag2 https://cnn.com"

my_hashtags=re.search(r'#\S+\s+\S+\s+#\S+', tweet_modified)
print(my_hashtags)

# (#\S+) matches a hashtag "#", followed by one or more non-whitespaces
#----------------------------------------
# \s+ matches one or more whitespaces
#----------------------------------------
# \s+\S+\s+: Basically matches the " and " in the tweet, 
# (note the preceding and following spaces).

# groups() will return all matched groups as a tuple:
print(my_hashtags.groups())

# The pattern with search() above is useful if you specifically wanted
# a pattern that has "hashtag+space(s)+and+space(s)+hashtag"
# If you want just to get all hashtags in a tweet, just use "findall"
my_hashtags=re.findall(r'(#\S+)', tweet)
print(my_hashtags)

my_url=re.findall(r'(https://\S+.\S+)', tweet)
print(my_url)

# Compile a pattern for reuse.
#------------------------------
# The "|" helps us match a hashtag or an URL (so if both exist,
# we capture BOTH)
p=re.compile(r'(#\S+|https://\S+.\S+)')
matches=re.findall(p, tweet) # 
print(matches)

# Using the paranthes to capture a group is useful
# if you wanted to substitute
new_tweet=re.sub(r'(#\S+)', '<HASHTAG>', tweet)
print(new_tweet)

new_tweet=re.sub(r'(#\S+)', '<HASHTAG>', tweet)
print(new_tweet)

# Add ?P<name> before a pattern to group by name
my_hashtags=re.search(r'(?P<first>#\S+)\s+\S+\s+(?P<second>#\S+)', tweet)
print(my_hashtags.group("first")) # whatever is in the first ()
print(my_hashtags.group("second")) # whatever is in the second ()

# Find all words with the character "s"
story="Samy told me an interesting story was airing on CBC last night..."
re.findall(r'\w+s\w+', story)

# Since \w* matches zero or more characters, we can get all words
# with "s" as follows:
re.findall(r'\w*s\w*', story)

# Well, almost! Let's ignore case with "re.I" to catch "Samy" as well.
re.findall(r'\w*s\w*', story, re.I)

