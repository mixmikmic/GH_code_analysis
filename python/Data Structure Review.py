my_list = [1,2,3,4,5,6,7]
#print(my_list)
print(my_list.reverse)
my_list.reverse()
#print(my_list)
# Note different methods permantly alter the list (but not all)
#print(my_list)
print(my_list)

a = 5
b = a

b = 7

print(id(a))
print(id(7))

copy_list = my_list
print(copy_list)

copy_list.append('hello!')

print(copy_list)
print(my_list)
print(id(copy_list))
print(id(my_list))
my_actual_copy = my_list[:]
print(my_actual_copy)
my_actual_copy.append('ok')
print(my_actual_copy)
print(my_list)

l = [1,2,3,4]
d = {'a':1}
d['a']

def my_function(name='Nick'):
    print('hello', name)
    
my_function()

def make_a_fresh_list():
    new_list = []
    for character in "Nick Tallant :)":
        new_list.append(character)
    return new_list

print(make_a_fresh_list())
print(make_a_fresh_list())

from pprint import pprint
#Intead of 
new_list = []
def not_a_fresh_list(new_list):
    for i in range(10):
        new_list.append('Python is great!')
    return new_list
pprint(not_a_fresh_list(new_list))
pprint(not_a_fresh_list(new_list))

# Pointer Example
my_dict = {'a': 1, 'b': 2}
my_other_dict = my_dict
my_other_dict['c'] = 3
print(my_other_dict)
print(my_dict)

# Source: https://www.opensecrets.org/members-of-congress/contributors?cid=N00004357&cycle=CAREER
Paul_Ryans_top_5 = {"Northwestern Mutual": "$271,450",
                    "BlackRock Inc": "$182,300",
                    "Blackstone Group": "$156,812",
                    "Blue Cross/Blue Shield":"$153,211",
                    "Koch Industries" :"$142,822"}
# hint - use replace method
Paul_Ryans_top_5['Koch Industries']

for company in Paul_Ryans_top_5:
    print(company)

type(Paul_Ryans_top_5.keys())

for company in Paul_Ryans_top_5.keys():
    print(company)

# note it is not values(), not values - this is because we are using a method!
for amount in Paul_Ryans_top_5.values():
    print(amount)

for tup in Paul_Ryans_top_5.items():
    print(tup)

for key, value in Paul_Ryans_top_5.items():
    print(key, "bribed Paul Ryan with", value)

# YOUR CODE HERE

# We can still iterate!
my_tuple = ('a','b',1,2,3)
for thing in my_tuple:
    print(thing)

# But we can't do this
my_tuple[0] = 'c'

my_set = {'a','b','c'}
for letter in my_set:
    print(letter)

my_list_2 = [0,1,0,1,0,1,0,1]
print(my_list_2)
print(set(my_list_2))

print(my_list, my_list[0])
print(my_dict, my_dict['a'])
print(my_tuple, my_tuple[0])
print('this is a string'[0])



