from pulp import *

#Create the problem
prob=LpProblem("Minimize Cost",LpMinimize)

#Define the data 
vehicle=["TV Station", "Cable Station 1","Cable Station 2"]

cost=[195,200,175]

maxmoney=[9000,10000,9000]

viewers=[50000,40000,30000]

advimpress=[7,4,5]

#Create the variables
TVb=LpVariable("TV Station broadcast",0,None, LpInteger)
C1b=LpVariable("Cable Station 1 broadcast",0,None, LpInteger)
C2b=LpVariable("Cable Station 2 broadcast",0,None, LpInteger)


#Objective Function is added
prob+=TVb*cost[0]+C1b*cost[1]+C2b*cost[2]

#Constraints are added
prob+=viewers[0]/advimpress[0]*TVb+viewers[1]/advimpress[1]*C1b+viewers[2]/advimpress[2]*C2b>=1000000
prob+=TVb*cost[0]<=maxmoney[0]
prob+=C1b*cost[1]<=maxmoney[1]
prob+=C2b*cost[2]<=maxmoney[2]

prob.solve()

print("Status :", LpStatus[prob.status])
print("")
for v in prob.variables():
    print(v.name,'=',v.varValue)
print("")
print('Minimun cost=', value(prob.objective))

