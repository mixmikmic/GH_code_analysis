some_string = "Alex is in Chicago, writing some code."
some_other_string = "John is in the park, feeding some birds."
and_another_string = "Sandra is in Austin, hiring some new people."
promise_last_string = "Jenny is in the newsroom, editing some stories."

name = 'Robby'
print('Where is %s?' % name)

age = 47
print('%s is %d years old.' % (name, age))

print('Where is {}?'.format(name))

print('{1} is {0} years old.'.format(age, name))

print("Where's {0}, that adorable {1}-year-old? Oh, there's {0}.".format(name, age))

names = ['Alex', 'John', 'Sandra', 'Jenny']
locs = ['Chicago', 'the park', 'Austin', 'an office']
actions = ['writing', 'feeding', 'hiring', 'editing']
stuff = ['code', 'birds', 'new people', 'stories']

for x in range(0, len(names)):
    print("{0} is in {1}, {2} some {3}.".format(names[x], locs[x], actions[x], stuff[x]))

names.append('Roger')
locs.append('Pasadena')
actions.append('baking')
stuff.append('cookies')

print("{0} is in {1}, {2} some {3}.".format(names[-1], locs[-1], actions[-1], stuff[-1]))

