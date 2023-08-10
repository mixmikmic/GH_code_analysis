#Solution to the problem:

from pulp import *

prob =LpProblem("Maximize Calories", LpMaximize)

objects=["Candy Bar","Sandwich","Juice Can","Apple"]

#Capture the information

Calories={objects[0]:90,
          objects[1]:130,
          objects[2]:100,
          objects[3]:40}

Protein={objects[0]:5,
          objects[1]:40,
          objects[2]:15,
          objects[3]:3}

Weigh={objects[0]:.25,
       objects[1]:.35,
       objects[2]:.32,
       objects[3]:.30}

Volume={objects[0]:.0005,
        objects[1]:.002,
        objects[2]:.00075,
        objects[3]:.0009}

#Create the variables

variables=LpVariable.dicts("Objects", objects,0,None, LpInteger)

#Objective Function is added
prob += lpSum([Calories[i]*variables[i] for i in objects])

#Constraints are added
prob+= lpSum([Protein[i]*variables[i] for i in objects]) >=200
prob+= lpSum([Weigh[i]*variables[i] for i in objects]) <= 10
prob+= lpSum([Volume[i]*variables[i] for i in objects])<= .0125

#Solution to the problem: 
prob.solve()
print ("Status:",LpStatus[prob.status])
print("")
for v in prob.variables():
    print (v.name,"=",v.varValue)
print("")
print("Total Calories =",value(prob.objective))

