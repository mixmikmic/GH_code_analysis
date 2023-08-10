import csv
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

class Player():
    def __init__(self, position, name, salary, points, value, team):
        self.self = self
        self.position = position
        self.name = name
        self.salary = salary
        self.points = points
        self.value = value
        self.team = team
        
    def __iter__(self):
        return iter(self.list)
    
    def __str__(self):
        return "{} {} {} {} {}".format(self.name,self.position,self.salary, self.points, self.team)

# This csv contains our predictions and salaries for each player. 
# We parse each row of the csv and convert it into a Player object.
with open('DKSalaries.csv', 'r') as data:
    reader = csv.reader(data)
    reader.next()
    players = []
    for row in reader:
        name = row[1]
        position = row[0]
        salary = int(row[2])
        points = float(row[4])
        value = points / salary 
        team = row[5]
        player = Player(position, name, salary, points, value, team)
        players.append(player)

def points_knapsack(players):
    budget = 50000
    current_team_salary = 0
    constraints = {
        'P':2,
        'C':1,
        '1B':1,
        '2B':1,
        '3B':1,
        'SS':1,
        'OF':3
        }
    
    counts = {
        'P':0,
        'C':0,
        '1B':0,
        '2B':0,
        '3B':0,
        'SS':0,
        'OF':0
        }
    
    players.sort(key=lambda x: x.points, reverse=True)
    team = []
    
    for player in players:
        nam = player.name
        pos = player.position
        if "/" in pos:
            pos=pos[:pos.index("/")]
        if "P" in pos:
            pos="P"
        sal = player.salary
        pts = player.points
        if counts[pos] < constraints[pos] and current_team_salary + sal <= budget:
            team.append(player)
            counts[pos] = counts[pos] + 1
            current_team_salary += sal

    return team

team = points_knapsack(players)
points = 0
salary = 0 
for player in team:
    points += player.points
    salary += player.salary
    print player
print "\nPoints: {}".format(points)
print "Salary: {}".format(salary)

def value_knapsack(players):
    budget = 50000
    current_team_salary = 0
    constraints = {
        'P':2,
        'C':1,
        '1B':1,
        '2B':1,
        '3B':1,
        'SS':1,
        'OF':3
        }
    
    counts = {
        'P':0,
        'C':0,
        '1B':0,
        '2B':0,
        '3B':0,
        'SS':0,
        'OF':0
        }
    
    players.sort(key=lambda x: x.value, reverse=True)
    team = []
    
    for player in players:
        nam = player.name
        pos = player.position
        if "/" in pos:
            pos=pos[:pos.index("/")]
        if "P" in pos:
            pos="P"
        sal = player.salary
        pts = player.points
        if counts[pos] < constraints[pos] and current_team_salary + sal <= budget:
            team.append(player)
            counts[pos] = counts[pos] + 1
            current_team_salary += sal

    return team

team = value_knapsack(players)
points = 0
salary = 0
for player in team:
    points += player.points
    salary += player.salary
    print player
print "\nPoints: {}".format(points)
print "Salary: {}".format(salary)

def improved_knapsack(players):
    budget = 50000
    current_team_salary = 0
    constraints = {
        'P':2,
        'C':1,
        '1B':1,
        '2B':1,
        '3B':1,
        'SS':1,
        'OF':3
        }
    
    counts = {
        'P':0,
        'C':0,
        '1B':0,
        '2B':0,
        '3B':0,
        'SS':0,
        'OF':0
        }
    
    players.sort(key=lambda x: x.value, reverse=True)
    team = []
    
    for player in players:
        nam = player.name
        pos = player.position
        if "/" in pos:
            pos=pos[:pos.index("/")]
        if "P" in pos:
            pos="P"
        sal = player.salary
        pts = player.points
        if counts[pos] < constraints[pos] and current_team_salary + sal <= budget:
            team.append(player)
            counts[pos] = counts[pos] + 1
            current_team_salary += sal
    
    players.sort(key=lambda x: x.points, reverse=True)
    for player in players:
        nam = player.name
        pos = player.position
        sal = player.salary
        pts = player.points
        if player not in team:
            pos_players = [ x for x in team if x.position == pos]
            pos_players.sort(key=lambda x: x.points)
            for pos_player in pos_players:
                if (current_team_salary + sal - pos_player.salary) <= budget and pts > pos_player.points:
                    team[team.index(pos_player)] = player
                    current_team_salary = current_team_salary + sal - pos_player.salary
                    break
    return team

team = improved_knapsack(players)
points = 0
salary = 0
p=[]
for player in team:
    points += player.points
    salary += player.salary
    p.append(str(player).split(" "))
    print player
print "\nPoints: {}".format(points)
print "Salary: {}".format(salary)



mlbDF=pd.read_csv('DKSalaries.csv')
mlbDF.head()

mlbDF.info()









