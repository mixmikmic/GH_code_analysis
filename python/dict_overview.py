character = {'name': 'Arthur',
             'weapon': 'Gold Sword',
             'armor': 'Steel Armor'}
character

# NOTE: dictionaries are unordered.

character['weapon']

character_list = ['Arthur', 'gold sword', 'steel armor']

character_list[0]

character_dict = {'name': 'Arthur', 'weapon': 'gold sword', 'armor': 'steel armor'}

character_dict['name']

ex_list1 = ['val1', 'val2']
ex_list2 = ['val2', 'val1']

print('Are these lists equivalent: ', ex_list1 == ex_list2)

# As noted previously... dictionaries are unordered.

ex_dict1 = {'key1': 'val1', 'key2': 'val2'}
ex_dict2 = {'key2': 'val2', 'key1': 'val1'}

print('Are these dicts equivalent: ', ex_dict1 == ex_dict2)

character['spells']

character['spells'] = ['fireball', 'lightning', 'summon dragon']

character['spells']

print(character.keys())

# the output of this is a view of the keys in character
# it might resemble a list, but it has subtle differences...

for key in character.keys():
    print(key)

print(character.values())

for v in character.values():
    print(v)

# What if you want both the keys and values (as pairs)

for k, v in character.items():
    print(k + ":\t", v)

character['weapon']

character['gold']

# .get() allows you to _get_ a default value back... but does not alter 
# the dictionary.

character.get('gold', 'You have no money')

character['gold']

# .setdefault() on the other hand, allows you to _set_ a dictionary value
# based on a default if the value does not already exist.

character.setdefault('gold', 0)

print(character['gold'])

# if the value exists already, the setdefault() function simply reads the 
# existing value.

character.setdefault('gold', 50)

# notice that character now has a value for the 'gold' key.

print(character)

equipped = {}
for item in ['armor', 'weapon', 'ring', 'helmet']:
    equipped[item] = input('What would you like for your ' + item + ': ')

print(equipped)

