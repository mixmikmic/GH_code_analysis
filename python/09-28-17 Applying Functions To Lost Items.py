regimentNames = ['Night Riflemen', 'Jungle Scouts', 'The Dragoons', 'Midnight Revengence', 'Wily Warriors']

# Create a variable for the 'for loop' result
regimentNamesCapitalized_f = []

# for every item in regimentNames
for i in regimentNames:
    # capitalize the item and add it to regimentNamesCapitalized_f
    regimentNamesCapitalized_f.append(i.upper())
    
# View the outcome
regimentNamesCapitalized_f

capitalizer = lambda x: x.upper()

regimentNamesCapitalized_m = list(map(capitalizer, regimentNamesCapitalized_f))
regimentNamesCapitalized_f

regimentNamesCapitalized_l = [x.upper() for x in regimentNames]; regimentNamesCapitalized_l

