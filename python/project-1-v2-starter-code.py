from pprint import pprint
import random
import numpy as np

class player:
    
    player_id = 0
    
    def __init__(self, name):
        player.player_id += 1
        self.player_id = player.player_id
        self.player_name = name
        self.time_played = 0.0
        self.player_pokemon = {}
        self.gyms_visited = []

David = player('David')
print(David)

gyms = ['reddit.com', 'amazon.com', 'twitter.com', 'linkedin.com', 'ebay.com', 'netflix.com', 'sporcle.com', 'stackoverflow.com', 'github.com', 'quora.com']

for x in range(2):
    David.gyms_visited.append(random.choice(gyms))

print(David.gyms_visited)

pokedex = {
    1:{
        'name':'charmander',
        'type':'fire',
        'hp':100,
        'attack':60,
        'defense':30,
        'special_attack':4,
        'special_defense':2,
        'speed':9,
    },
    
    2:{
        'name':'squirtle',
        'type':'water',
        'hp':100,
        'attack':50,
        'defense':50,
        'special_attack':3,
        'special_defense':5,
        'speed':6,
    },
    
    3:{
        'name':'bulbasaur',
        'type':'poison',
        'hp':100,
        'attack':40,
        'defense':50,
        'special_attack':3,
        'special_defense':4,
        'speed':3,
    }
}

pprint(pokedex)


players = {
    David.player_id:{
        'player_name':David.player_name,
        'time_played':David.time_played,
        'player_pokemon':David.player_pokemon,
        'gyms_visited':David.gyms_visited
    }
}

pprint(players)

John = player('John')
John.gyms_visited.extend(['alcatraz', 'pacific_beach'])

players.update({
    John.player_id:{
        'player_name':John.player_name,
        'time_played':John.time_played,
        'player_pokemon':John.player_pokemon,
        'gyms_visited':John.gyms_visited
    }
})

pprint(players)

for x, v in pokedex.iteritems():
    if v['name'] == 'squirtle':
        players[1]['player_pokemon'].update({x:v})
    if v['name'] == 'charmander':
        players[2]['player_pokemon'].update({x:v})
    if v['name'] == 'bulbasaur':
        players[2]['player_pokemon'].update({x:v})

pprint(players)


for gym in gyms:
    for player, player_details in players.iteritems():
        if gym in player_details['gyms_visited']:
            print(player_details['player_name'] + " has visited " + gym + ".")

# That loop ran 20 times
# It will run N squared times
for player, player_details in players.iteritems():
        for gym in player_details['gyms_visited']:
            if gym in gyms:
                print(player_details['player_name'] + " has visited " + gym + ".")


def player_power(players, pokedex, player_id):
    
    player_pokemon = players[player_id]['player_pokemon']
    power = 0

    for pokemon_id in player_pokemon.keys():
        if pokemon_id in pokedex.keys():
            power += pokedex[pokemon_id]['attack']
            power += pokedex[pokemon_id]['defense']
    
    print(players[player_id]['player_name'] + "'s power is " + str(power))
    return power

for id in players.keys():
    print(player_power(players, pokedex, id))

# Code to read in pokedex info
raw_pd = ''
pokedex_file = 'pokedex_basic.csv'
with open(pokedex_file, 'r') as f:
    
    # the pokedex string is assigned to the raw_pd variable
    raw_pd = f.read()
    list_pd = raw_pd.split('\n')
    # initialize most outer list of pokedex
    pokedex_matrix = []
    
    # other than the first which is the header, each sublist_string represents one pokemon
    for sublist_string in list_pd:
        
        #remove quotes from each cell
        string_trimmed = sublist_string.replace('"','')
        # each pokemon will represent a list 
        sublist = string_trimmed.split(',')
        # Initialize inner list of pokedex
        pokedex_row = []
        
        # Go through each attribute value of the pokemon and check for
        # numeric and empty cells, converting them into floats and -1 respectively
        for attribute in sublist:
            if attribute.isdigit():
                attribute = float(attribute)
            elif attribute == '':
                attribute = -1
            # add value into new pokemon list
            pokedex_row.append(attribute)
        # add new pokemon list into pokedex
        pokedex_matrix.append(pokedex_row)
    
    f.close()
    
    #output first three rows from Matrix
    pprint(pokedex_matrix[:3])



# transform raw string into matrix
pokedex_matrix = [
        [-1 if string.replace('"','') == '' else float(string.replace('"','')) if string.replace('"','').isdigit() else string.replace('"','') for string in sublist_string.split(',')]
    for sublist_string in raw_pd.split('\n')]

pprint(pokedex_matrix[:3])

# function will regenerate unique ID for every pokemon
# due to duplicates in CSV file
def generate_pokedex(parsed_pokedex):
    pokedex_dict = {}
    attributes = parsed_pokedex[0]
    # sort pokedex by attribute total
    sorted_pokedex = sorted(parsed_pokedex[1:], key=lambda x: x[3])
    new_id = 0.0
    
    for pokemon in sorted_pokedex:
        # every pokemon will take on a new ID
        new_id += 1
        for index, (header, value) in enumerate(zip(attributes, pokemon)):
            # first value in list is pokemon id
            if index == 0:
                pokedex_dict.update({new_id:{}})
            else:
                pokedex_dict[new_id].update({header:value})
    
    return pokedex_dict

pokedex = generate_pokedex(pokedex_matrix)
pprint(pokedex[800])

filter_options = {
    'Attack':   25,
    'Defense':  30,
    'Type':     'Electric'
}

# Assumption is pokedex and filter are all dictionaries
# Assumption is all filter conditions must be met
def filtered_pokedex(pokedex_data, filter=None):
    
    if filter == None:
        return "No filter specified"
    else:
        #initialize filter result array
        filter_result = []
        
        #loop through each pokemon in pokedex
        for pokemon_id, pokemon_details in pokedex_data.iteritems():
            filter_values_string = []
            filter_values_float = []
            pokemon_values_string = []
            pokemon_values_float = []
            
            #Go through every attribute of the pokemon
            for key, value in pokemon_details.iteritems():
                # if the current attribute is found in the filter, capture and store it
                if key in filter.keys():
                    # Store both values of attributes separately but in the same position for easy comparison later
                    # string and float values have different comparison operators, thus we segregate them
                    if isinstance(pokemon_details[key], float):
                        filter_values_float.append(filter[key])
                        pokemon_values_float.append(pokemon_details[key])
                    if isinstance(pokemon_details[key], str):
                        filter_values_string.append(filter[key])
                        pokemon_values_string.append(pokemon_details[key])
            
            #once collected all necessary values, compare the filter values with the pokemon values
            string_filter_result = [True if fv == pv else False for fv, pv in zip(filter_values_string, pokemon_values_string)]
            float_filter_result = [True if pv >= fv else False for fv, pv in zip(filter_values_float, pokemon_values_float)]
            #all the conditions must be met, e.g. float values must be equal or bigger, string values must be identical
            #if all met, capture and store this pokemon in the filter result array
            if False not in string_filter_result and False not in float_filter_result:
                filter_result.append(pokemon_details)
            
    return filter_result

pprint(filtered_pokedex(pokedex, filter=filter_options))

Total = []

for pokemon_id, pokemon_details in pokedex.iteritems():
    Total.append(pokemon_details['Total'])

Total_mean = np.mean(Total)
Total_std = np.std(Total)

print("Population mean for 'Total' is " + str(Total_mean))
print("Population standard deviation for 'Total' is " + str(Total_std))

Character = []

for pokemon_id, pokemon_details in pokedex.iteritems():
    Character.append(pokemon_details['Name'])

np_Total = np.array(Total)
np_Character = np.array(Character)

pprint(np_Character[np_Total>Total_mean+(3*Total_std)])

# assumption is probability of encountering any two pokemon on one visit
# is independent of each other, independent variables, 1 trial = 1 visit to gym, each trial sum of all probability = 1
# probability and power must have inverse relationship
# must ensure all probabilities of pokemon adds up to one
# must be more dynamic taking into account new pokemon added into pokedex
def gen_pokemon_prob(pokedex_number):
    # get inverted power of each pokemon
    total = [[pokemon_id, 1/values['Total']] for pokemon_id, values in pokedex.iteritems()]
    # sort this list from small to big by inverted power
    total = sorted(total, key=lambda x: x[1])
    # divide each inverted power by sum total of inverted values so that probability always adds up to one when you add 
    pokemon_prob = {x[0]:x[1]/sum([y[1] for y in total])*100 for x in total}
    return pokemon_prob[pokedex_number]

print(pokedex[1]['Name'], gen_pokemon_prob(1))
print(pokedex[800]['Name'], gen_pokemon_prob(800))

# Scale the power column and return pokemon dictionary with power
# and corresponding probability
def gen_pokemon_power_prob_dict(pokedex_dict):
    inverted_total = [[pokemon_id, 1/values['Total']] for pokemon_id, values in pokedex_dict.iteritems()]
    power_range = np.ptp(Total)
    pokemon_power_prob = {x[0]:[pokedex_dict[x[0]]['Total']/power_range, x[1]/sum([y[1] for y in inverted_total])*100] for x in inverted_total}
    return pokemon_power_prob

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pkmon_pw_pb_dict = gen_pokemon_power_prob_dict(pokedex)
sns.set(color_codes=True)
power_arr = [p[0] for p in pkmon_pw_pb_dict.itervalues()]
prob_arr = [p[1] for p in pkmon_pw_pb_dict.itervalues()]
plt.xlabel('Power')
plt.ylabel('Probability in %')
plt.scatter(x=power_arr, y=prob_arr)



