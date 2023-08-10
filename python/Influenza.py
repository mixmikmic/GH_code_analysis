import yaml

with open("/Users/sahuguet/Documents/Dev/Influenza/legislators-current.yaml", 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

for people in data:
    name = people['name']['official_full'].replace('"','\\"')
    role = people['terms'][-1]['type']
    party = people['terms'][-1]['party']
    if role == 'sen':
        print """CREATE (:Person:Senator { name: "%s", party: "%s" })""" % (name, party)
    elif role == 'rep':
        print """CREATE (:Person:Representative { name: "%s", party: "%s" })""" % (name, party)

