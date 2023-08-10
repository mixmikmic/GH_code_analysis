# Let's parse the data from the last mission as an example.
# First, we open the wait times file from the last mission.
f = open("crime_rates.csv", 'r')
data = f.read()
rows = data.split('\n')
full_data = []
for row in rows:
    split_row = row.split(",")
    full_data.append(split_row)

weather_data = []
f = open("la_weather.csv", 'r')
data = f.read()

rows = data.split('\n')
full_data = []
for row in rows:
    split_row = row.split(",")
    weather_data.append(split_row)
    
print(weather_data[:10])

# The "days" column in our data isn't extremely useful for our task, so we need to just grab the second column, with the weather.
# We looped over lists before, and this is how we will extract the second column.
lolist = [[1,2],[3,4],[5,6],[7,8]]
second_column = []
for item in lolist:
    # Each item in lolist is a list.
    # We can get just the second column value by indexing the item.
    value = item[1]
    second_column.append(value)

# second_column is now a list containing only values from the second column of lolist.
print(second_column)

# Let's read in our weather data again.
weather_data = []
f = open("la_weather.csv", 'r')
data = f.read()
rows = data.split('\n')
for row in rows:
    split_row = row.split(",")
    weather_data.append(split_row)

weather_column = []

for row in weather_data:
    val = row[1]
    weather_column.append(val)
    
print(weather_column)

weather = weather_column

# In order to make it easier to use the weather column that we just parsed, we're going to automatically include it from now on.
# It's been specially added before our code runs.
# We can interact with it normally -- it's a list.
print(weather[0])

count = len(weather)
print(count)

# Let's practice with some list slicing.
a = [4,5,6,7,8]
# New list containing index 2 and 3.
print(a[2:4])
# New list with no elements.
print(a[2:2])
# New list containing only index 2.
print(a[2:3])

slice_me = [7,6,4,5,6]

slice1 = slice_me[2:4]
slice2 = slice_me[1:2]
slice3 = slice_me[3:]

print(slice1, slice2, slice3)

new_weather = weather[1:]
print(new_weather)

# We can make a dictionary with curly braces.
dictionary_one = {}

# The we can add keys and values.
dictionary_one["key_one"] = 2
print(dictionary_one)

# Keys and values can be anything.
# And dictionaries can have multiple keys
dictionary_one[10] = 5
dictionary_one[5.2] = "hello"
print(dictionary_one)

dictionary_two = {
    "test": 5,
    10: "hello"
}

print(dictionary_two)

dictionary_one = {}
dictionary_one["test"] = 10
dictionary_one["key"] = "fly"
# We can retrieve values from dictionaries with square brackets.
print(dictionary_one["test"])
print(dictionary_one["key"])

dictionary_two = {}
dictionary_two["key1"] = "high"
dictionary_two["key2"] = 10
dictionary_two["key3"] = 5.6

a, b, c = dictionary_two["key1"], dictionary_two["key2"], dictionary_two["key3"]
print(a, b, c)

# We can define dictionaries that already contain values.
# All we do is add in keys and values separated by colons.
# We have to separate pairs of keys and values with commas.
a = {"key1": 10, "key2": "indubitably", "key3": "dataquest", 3: 5.6}

# a is initialized with those keys and values, so we can access them.
print(a["key1"])

# Another example
b = {4: "robin", 5: "bluebird", 6: "sparrow"}
print(b[4])

c = {
    7: "raven",
    8: "goose",
    9: "duck"
}

d = {
    "morning": 9,
    "afternoon": 14,
    "evening": 19,
    "night": 23
}

print(c, d)

# We can check if values are in lists using the in statement.
the_list = [10,60,-5,8]

# This is True because 10 is in the_list
print(10 in the_list)

# This is True because -5 is in the_list
print(-5 in the_list)

# This is False because 9 isn't in the_list
print(9 in the_list)

# We can assign the results of an in statement to a variable.
# Just like any other boolean.
a = 7 in the_list

list2 = [8, 5.6, 70, 800]

c, d, e = 9 in list2, 8 in list2, -1 in list2

print(c, d, e)

# We can check if a key is in a dictionary with the in statement.
the_dict = {"robin": "red", "cardinal": "red", "oriole": "orange", "lark": "blue"}

# This is True
print("robin" in the_dict)

# This is False
print("crow" in the_dict)

# We can also assign the boolean to a variable
a = "cardinal" in the_dict
print(a)

dict2 = {"mercury": 1, "venus": 2, "earth": 3, "mars": 4}

b = "jupiter" in dict2
c = "earth" in dict2

print(b, c)

# The code in an else statement will be executed if the if statement boolean is False.
# This will print "Not 7!"
a = 6
# a doesn't equal 7, so this is False.
if a == 7:
    print(a)
else:
    print("Not 7!")

# This will print "Nintendo is the best!"
video_game = "Mario"
# video_game is "Mario", so this is True
if video_game == "Mario":
    print("Nintendo is the best!")
else:
    print("Sony is the best!")

season = "Spring"

if season == "Summer":
    print("It's hot!")
else:
    print("It might be hot!")

# We can count how many times items appear in a list using dictionaries.
pantry = ["apple", "orange", "grape", "apple", "orange", "apple", "tomato", "potato", "grape"]

# Create an empty dictionary
pantry_counts = {}
# Loop through the whole list
for item in pantry:
    # If the list item is already a key in the dictionary, then add 1 to the value of that key.
    # This is because we've seen the item again, so our count goes up.
    if item in pantry_counts:
        pantry_counts[item] = pantry_counts[item] + 1
    else:
        # If the item isn't already a key in the count dictionary, then add the key, and set the value to 1.
        # We set the value to 1 because we are seeing the item, so it's occured once already in the list.
        pantry_counts[item] = 1
print(pantry_counts)

us_presidents = ["Adams", "Bush", "Clinton", "Obama", "Harrison", "Taft", "Bush", "Adams", "Wilson", "Roosevelt", "Roosevelt"]

###################
# answer #1
us_president_counts = {}

for p in us_presidents:
    if p not in us_president_counts:
        us_president_counts[p] = 0
    us_president_counts[p] += 1

####################
# answer #2
us_president_counts = dict([(_, 0) for _ in set(us_presidents) ])

for p in us_presidents:
    us_president_counts[p] += 1



print(us_president_counts)

weather = weather_column[1:]

from collections import defaultdict
weather_counts = defaultdict(lambda: 0)

for w in weather:
    if w not in weather_counts:
        weather_counts[w] = 0
    weather_counts[w] += 1
    
print(weather_counts)



