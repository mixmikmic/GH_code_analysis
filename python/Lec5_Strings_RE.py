s1 = 'String One'
s2 = "string two"

s3 = '''Multi
line
string'''
s4 = """Multi
line
string"""
s3 == s4 # The quotes chosen don't matter

print(s3) # Note the way this string prints

print(s1[2])
s1[0:6]

for l in s1:
    print(l, end = "  ")

s1.upper() # Convert to uppercase

s1.lower() # Convert to lowercase

print(s2)
print(s2.title()) # Convert to titlecase

s2.capitalize() # Capitalize (make first letter capital)

print(s1)
print(s1.swapcase()) # Change the case

s = "    some text            "
s.strip()

s.lstrip()

s.rstrip()

s = "++++++pure text+++++"
s.strip("+")

s.rstrip("+")

s = "small piece of text"
s.center(30) # The argument is the overall length 
             # of the resulting string

s.center(30,"^")

s.ljust(40)

# You can also specify the string directly
# instead of defining a variable
"Python is sooo cool".rjust(25,"☯")

x = 777
str(x).zfill(10)

s = "A yellow python is prettier than a black python."
s.find("python")

s.index("python")

s.find("anaconda")

s.index("anaconda")

s.rfind("python")

s.find("python", 15)

s.find("python", 15, 30)

s.startswith("A blue")

s.endswith("python.")

s.replace("python", "anaconda")

s.partition("python")

s.rpartition("python")

s.split()

s.split("y")

Zen = """In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch."""

Zen.splitlines() # Try passing True as an argument
                 # to include newline characters in the result

concatstring = " "
iterstring = ("A","fine","day")
# concatstring = " - ahem - "
concatstring.join(iterstring)

lines = ["Gaudeamus igitur.", 
         "Iuvenes dum sumus.",
         "Post iucundam iuventutem.", 
         "Post molestam senectutem."]
result = "\n".join(lines)
result

RB = """Gin a body meet a body
Comin thro' the rye,
Gin a body kiss a body,
Need a body cry?"""

RB.count("body")

RB.count("body",23,67)

"body" in RB

"somebody" in RB

print("abc".isalnum())
print("123".isalnum())
print("a1B3".isalnum())
print("".isalnum()) # empty string is not
print("ab1@".isalnum())
print("45.2".isalnum())

print("123".isdigit())
print("abc".isdigit())
print("123.5".isdigit())
print("000556".isdigit())
print("VI".isdigit())

get_ipython().magic('reset')
import re

text = "The Ring, which Gollum referred to as 'my precious' or 'precious', extended his life far beyond natural limits."
p = re.compile("precious") # p is a pattern object
m = p.search(text) # m is a match object
x,y = m.start(),m.end() # starting and ending positions 
                        # of first match
print(text[x:]) 
print(text[y:])
l = p.findall(text) # find all matches and return them in a list
# The result is trivial in this case
l

m = p.match(text)
if m:
    print(m)
else:
    print("No match!")

text = "A bee can see a zee from afar"
p = re.compile("[bs]ee")
# p = re.compile("[^bs]ee")
itr = p.finditer(text) # Produces an iterable object
for e in itr:
    print("The word '%s' can be found at position %d."%(e.group(), e.start()))

text = """112 blue bottles hanging on the wall,
10 green bottles hanging on the wall,
9 red bottles hanging on the wall,
1 green bottle hanging on the wall."""
p = re.compile("([0-9]+ (?:blue|green) bottle[s]?)")
# Here (?:blue|green) is a non-capturing group
# The simpler (blue|green) will capture these matches separately

# Try also:
# p = re.compile("[0-9]+.*bottle[s]?")
L = p.findall(text)
L

print("this is a regular \n string")
print(r"this is a raw \n string")

plates = """A police officer with badge number PO31254 wrote 
speeding tickets for vehicles with licence plates 
CA6542HP, 234GH856, C1234AA and B4455TK. Another officer, 
holding badge numbered CA98765 and born on 11.11.1970, 
wrote tickets for vehicles CO7391KK, CA3571ET, T1213MA and CB6534EH."""
p = re.compile(r"C[AB]?\d{4}[A-Z]{2}")
L = p.findall(plates)
L

