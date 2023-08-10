import csv

file = 'data/TXTran.csv'
file2 = 'data/TXLeg.csv'

newFile = open(file)
newReader = csv.reader(newFile)
newData = list(newReader)

travel = []
leg = []

for row in newData[1:]:
    travel.append(row)

legFile = open(file2)
legReader = csv.reader(legFile)
legData = list(legReader)

for row in legData[1:]:
    leg.append(row)

from fuzzywuzzy import fuzz

travelFound = []

for item in travel:
    itemname = item[5] + ' ' + item[6]
    found = False
    for person in leg:
        personname = person[2] + ' ' + person[4]
        if itemname == personname or itemname == person[1]:
            found = True
            item.extend([person[0],person[5],person[6],person[7],person[8],person[9],personname])
            travelFound.append(item)
    if found == False:
        for person in leg:
            if fuzz.token_sort_ratio(itemname, personname) > 80:
                found = True
                item.extend([person[0],person[5],person[6],person[7],person[8],person[9],personname])
                print(itemname + ' fuzzy match ' + personname)
                travelFound.append(item)
    # if found == False:
        # print('No match found for ' + itemname)

print(travelFound[5])

import networkx as nx
G=nx.MultiDiGraph()

for row in travelFound:
    G.add_node(row[2], name=row[3], role="Lobbyist")
    G.add_node(row[9], name=row[-1], role=row[10])
    G.add_edge(row[2], row[9], gift="Travel", year=row[1], detail=row[8])

G.nodes(data=True)[:15]

G.is_directed()

G.node['TXL000211']

nx.degree(G)

H = nx.Graph(G)
answer = nx.connected_components(H)
for i in answer:
    print(i)

# This is the output. The lines after this are just experiments.

nx.readwrite.write_gexf(G,"lobby.gexf")

# I tried making this CSV to output for graphing. But this isn't that useful because it puts all the 
# data on the edges, none on the nodes, so I didn't end up using it for anything.

outputFile = open('StateLobbyistTravel.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['Record', 'Year', 'Source', 'LobbyName', 'Target','LegName','Party'])
for row in travelFound:
    outputWriter.writerow([row[0],row[1],row[2],row[3],row[9],row[-1],row[10]])
outputFile.close()

# Testing to see how many records there are where the legislator's party is unknown because OpenStates doesn't provide
# that info after they leave office.

noParty = []

for record in travel:
    if len(record) > 9:
        if record[10] == '':
            noParty.append(record)
        
len(noParty)

# Here's where I tried using Neo4Jj

from py2neo import Graph

graph = Graph()
graph.delete_all()

# This works to add nodes into a Neo4j database, but not relationships. Don't know why.

from py2neo import Node, Relationship

for record in travelFound:
    itemname = record[5] + ' ' + record[6]
    a = graph.merge(Node("Lobbyist", id=record[2], name=record[3]))
    b = graph.merge(Node("Legislator", id=record[9], name=itemname, party=record[10]))
    # Two attempted solutions below. neither of them is working.
    ab = graph.create(Path(a, "Travel", b))
    # graph.create(Relationship(a, b))

