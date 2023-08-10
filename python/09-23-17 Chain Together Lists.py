from itertools import chain

# Create a list of allies
allies = ['Spain', 'Germany', 'Namibia', 'Austria']

# Create a list of enemies
enemies = ['Mexico', 'United Kingdom', 'France']

# For each country in allies and enemies
for country in chain(allies, enemies):
    print(country)

