network1 = Graph({'Alice':['Bob'], 'Bob':['Charles'], 'Charles':['Dave'], 'Dave':['Ed']})
network1.show()

network2 = Graph({'Rob':['Steve', 'Ty', 'Ursula','Walt', 'Zeb'], 'Steve':['Ty', 'Zeb'], 'Ty':['Steve', 'Ursula'],
                  'Ursula': ['Ty', 'Walt'], 'Walt':['Ursula', 'Zeb'], 'Zeb':['Walt', 'Steve']})
network2.show()

g = graphs.RandomGNP(8, 0.4)   # Random graph with 8 nodes, 0.4 probability of two nodes being connected
g.show()

g.diameter()



