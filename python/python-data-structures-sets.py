# Define a simple set using { } curly braces and unique, homogenous, comma separated values
simple_set = {'Red', 'Green', 'Blue'}
simple_set

# Create a empty set
empty_set = set()
empty_set

# Create set from a string, which in turn is a list of characters
alphabet_set = set('a quick brown fox jumpled over a lazy dog')
alphabet_set

# Define a set using a list, eliminating duplicates in the process
popular_games_list = ['Call of Duty', 'Final Fantasy', 'Battlefield', 'Witcher', 'Final Fantasy', 'Witcher']
popular_games = set(popular_games_list)
popular_games

# Define a set using short form
owned_games = set(['Destiny', 'Battlefield', 'Fallout'])
owned_games

# Membership test
print('Fallout' in owned_games)
print('Fallout' in popular_games)

# Set difference
popular_not_owned = popular_games - owned_games
popular_not_owned

# Set intersection
popular_owned = popular_games & owned_games
popular_owned

# Set symmetric difference
popular_owned_unique = popular_games ^ owned_games
popular_owned_unique

# Set union
all_games = popular_games | owned_games
all_games

