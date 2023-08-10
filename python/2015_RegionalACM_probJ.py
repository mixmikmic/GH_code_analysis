def problem_j():
    def is_it_here(the_list, the_city):
        for k in xrange(len(the_list)):
            if the_list[k] == the_city:
                return True
        return False
    trips = int(raw_input())#how many trips
    for k in xrange(trips):
        cities = []
        stops = int(raw_input())#how many stops
        for j in xrange(stops):
            this = raw_input()
            if is_it_here(cities,this) == False:
                cities.append(this)
        print len(cities)

cities = []
n = int(raw_input())
for i in range(n):
    num_cities = int(raw_input())
    for j in range(num_cities):
        cities.append(raw_input())
    cities = set(cities)
    print(len(cities))
    cities = []

Num_Test_Cases = int(raw_input().strip())

for i in range(Num_Test_Cases):
    num_trips = int(raw_input().strip())
    set_cities = set()
    for i in range(num_trips):
        set_cities.add(raw_input())
    print len(set_cities)

nc = int(raw_input())
while(nc > 0):
    nc = nc - 1
    s = set()
    for i in range(int(raw_input())):
        s.add(raw_input())
    print len(s)

Num_Test_Cases = int(raw_input().strip())

for i in range(Num_Test_Cases):
    num_trips = int(raw_input().strip())
    set_cities = set()
    for i in range(num_trips):
        set_cities.add(raw_input())
    print len(set_cities)

# acm2015_J.py

def problem_J():
    T = int(raw_input().strip())
    visit = []
    for t in xrange(T):
        n = int(raw_input().strip())
        cities = set()
        for i in xrange(n):
            c = raw_input().strip()
            cities.add(c)
        visit.append(len(cities))
    for i in xrange(len(visit)):
        print visit[i]
    

