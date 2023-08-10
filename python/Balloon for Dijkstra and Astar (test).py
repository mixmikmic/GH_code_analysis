import os
import pandas as pd
citypath = os.path.expanduser('~wesle\Desktop\contest\data\CityData.csv')
city=pd.read_csv(citypath)
city 

T=pd.read_csv("C:\\Users\\wesle\\Desktop\\contest\\data\\In_situMeasurementforTraining_201712.csv")

dayone=T[T.date_id==1]
# xid 1-548 yid 1-421 hour 3-20 day 1-5
dayone.head()

class Vertex:
    def __init__(self,key):
        # key is the tuple
        self.id = key
        # each vertex has four neighbours and itself 
        # store in dictionary
        self.connectedTo = {}
        self.weather ={}
        
    def setWeather(self,timeid,wind):
        self.weather[timeid] = wind
  
    def addNeighbor(self,nbr,weight=1):
        #nbr is a vertex
        self.connectedTo[nbr.id] = weight
        # when I store the information about the connectivity, diction is used, key is tuple,obj is weight
    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([key for key in self.connectedTo])

class Graph:
    def __init__(self,max_x,max_y):
        self.vertList    = {}
        self.numVertices = max_x*max_y
        self.length      = max_y
        self.width       = max_x
        for x in range(1,max_x+1):
            for y in range(1,max_y+1):
                self.vertList[(x,y)] = Vertex((x,y))
                # each key is a tuple and the objects are Vertex obj
    
    def makegrid(self):
        ''' make connection among each vertex'''
        for vertice in self:
            #left 
            if vertice.id[0]>=2:
                left=(vertice.id[0]-1,vertice.id[1])  
                try:
                    self.addEdge(vertice.id,left)
                except ValueError as e:
                    print(e)
                    pass


            #right
            if vertice.id[0]<=self.width-1:
                right=(vertice.id[0]+1,vertice.id[1])             
                try:
                    self.addEdge(vertice.id,right)
                except ValueError as e:
                    print(e)
                    pass

            # up
            if vertice.id[1]<=self.length-1:
                up=(vertice.id[0],vertice.id[1]+1)             
                try:
                    self.addEdge(vertice.id,up)
                except ValueError as e:
                    print(e)
                    pass

            # down
            if vertice.id[1]>=2:
                down=(vertice.id[0],vertice.id[1]-1)             
                try:
                    self.addEdge(vertice.id,down)
                except ValueError as e:
                    print(e)
                    pass
    
    def getVertexNeighbour(self,v):
        if v in self:
            return set(self.vertList[v].connectedTo.keys())
        else:
            raise TypeError("it is not in the graph")
   
    def __contains__(self,n):
        return n in self.vertList
    
    def __iter__(self):
        return iter(self.vertList.values())
    
    def addEdge(self,f,t,cost=1):
        if f not in self.vertList:
            raise ValueError("f is not in the graph")
        if t not in self.vertList:
            raise ValueError("t is not in the graph")
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        
    def deleteEdge(self,f,t):
        if f not in self.vertList:
            raise ValueError("f is not in the graph")
        if t not in self.vertList:
            raise ValueError("t is not in the graph")
        # pairwise delete
        del self.vertList[f].connectedTo[t] 
        del self.vertList[t].connectedTo[f]
    
     # need to make obstacles
     # disconnect the dots with its neighbour
    def disconnect(self,thisvertex):
        neighbour=[key for key in self.vertList[thisvertex].connectedTo]
        for i in neighbour:
            self.deleteEdge(thisvertex,i)
            
    def printgraph(self):
        for myvertice in self:
            print(myvertice)
            
    def isdisconnect(self,v):
        # if all the neighbour is not reachable 
        if self.getVertexNeighbour(v) ==set():
            return True
        else: return False

def Dijkstra(graph,start,end):
    '''find the shortest path between start and end if there is no path return False else return the path '''
    if graph.isdisconnect(start):
        raise TypeError('The root of the shortest path tree cannot be reached')
    if graph.isdisconnect(end):
        raise TypeError('The target of the shortest path cannot be reached') 
     
    # initializing all vertex are set to be unvisited except for the start
    unvisited=set(graph.vertList.keys())
    visited={start}
    unvisited.remove(start)
    dist=dict()
    path=dict()
    for e in unvisited:
        dist[e]=float("inf") # infinite means not reachable
        path[e]=-1
    for n in graph.getVertexNeighbour(start):
        dist[n]=1
        path[n]=start
    dist[start]=0
    
    while unvisited is not None:
        #pick unvisited closest node 
        subsetdist={k: dist[k] for k in dist.keys() & unvisited}
        newnode=min(subsetdist,key=subsetdist.get)
        # move it from unvisited to visited
        if newnode==end:
            break
        visited.add(newnode)
        
        unvisited.remove(newnode)

        # update the distance
        for n in graph.getVertexNeighbour(newnode):
            if dist[newnode]+1<=dist[n]:
                dist[n]=dist[newnode]+1
                path[n]=newnode
    # print out the path
    out=[end]
    temp=path[end]
    while temp != start:
        out.append(temp)
        temp=path[temp]
    out.append(start)
    return out

from collections import deque

class AStar():
    def __init__(self,graph,dangerzone):
        self.graph=graph
        # danger zone is a diction with index from 4 to 20  each element is a list of tuples which is danger for the balloon
        self.dangerzone=dangerzone

    def distBetween(self,current,neighbor):
    # since it is in the grid the distance between the neighbour is always 1
        return 1

    def heuristicEstimate(self,start,goal):
        # we use manhattan distance
        dx=abs(start[0]-goal[0])
        dy=abs(start[1]-goal[1])
        return (dx+dy)

    def neighborNodes(self,current):
        return set(self.graph.vertList[current].connectedTo.keys())
    
    def reconstructPath(self,cameFrom,goal):
        '''this is going to print the path'''
        path = deque()
        node = goal
        path.appendleft(node)
        while node in cameFrom:
            node = cameFrom[node]
            path.appendleft(node)
        return path
    
    def getLowest(self,openSet,fScore):
        lowest = float("inf")
        lowestNode = None
        for node in openSet:
            if fScore[node] < lowest:
                lowest = fScore[node]
                lowestNode = node
        return lowestNode

    def updategraph(self,dzlist):
        self.graph.makegrid()
        for e in dzlist:
            self.graph.disconnect(e)
    
    def aStar(self,start,goal):
        cameFrom = {}
        openSet = set([start])
        closedSet = []# this is the list we visited
        gScore = {} # this one store the distance from starting point to the point in closedSet
        fScore = {} 
        gScore[start] = 0
        fScore[start] = gScore[start] + self.heuristicEstimate(start,goal)
        # I would like to add a clock to count down the steps 
        time=1
        

            
        
        while len(openSet) != 0:

            current = self.getLowest(openSet,fScore)
            
            if current == goal:
                return self.reconstructPath(cameFrom,goal)
            openSet.remove(current)
            closedSet.append(current)
            
            
            
            for neighbor in self.neighborNodes(current):
                tentative_gScore = gScore[current] + self.distBetween(current,neighbor)
                if neighbor in closedSet and tentative_gScore >= gScore[neighbor]: # if it is visited and the cost is bigger than original ones 
                    continue# we don't store this neighbour and do nothing
                    
                if neighbor not in closedSet or tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = gScore[neighbor] + self.heuristicEstimate(neighbor,goal)
                    # update a new way to get to this point
                    # if it has not been visited we add it to the openset
                    if neighbor not in openSet:
                        openSet.add(neighbor)
            
            time =time+1
           
            if time % 30 ==0: 
                hour=time/30+3 # hour start from 3 to 20 
                print(hour)
                self.updategraph(self.dangerzone[hour])
                # reset the openSet it only contains the last one's neighbour
                if self.graph.isdisconnect(current):
                    print("go to wrong way have to go back")
                    backstep=1
                    closedSet.pop()
                    previous=closedSet.pop()
                    while(self.graph.isdisconnect(previous)):
                        previous=closedSet.pop()
                        backstep =backstep+1
                    current=previous
                    closedSet.append([current]*backstep)
                    
                openSet.clear()
                for neighbor in self.neighborNodes(current):
                    print(neighbour)
                    openSet.add(neighbor)
            
        
        return 0

# get the danger zone
def makedangerzone(df):
    # df contains only one day's data
    dangerzone=dict()
    for hour in set(df.hour.unique()):
#         print(hour)
        temp=df[df.hour==hour][df.wind>=15].iloc[:,0:2]
        dangerzone[hour]=[tuple(x) for x in temp.values]
    return dangerzone
dangerzone=makedangerzone(dayone)

city


import gc
def drawdanger(df,hour=3):
    danger3= pd.DataFrame(dangerzone[hour])
    danger3=danger3.rename(index=str,columns={0:"x",1:"y"})
    gc.collect()
   
    print(ggplot(aes(x="x",y="y"),data=danger3)+geom_point())

drawdanger(dangerzone)

drawdanger(dangerzone,hour=4)

wholegraph=Graph(max_x=548,max_y=421)

wholegraph.makegrid()

wholeastar=AStar(wholegraph,dangerzone)

start

goal

df

wholeastar.aStar(start,goal)
temproute# updating the graph is impossible  

temproute=pd.DataFrame(list(temproute))
temproute.head(10)

temproute=temproute.rename(index=str,columns={0:"x",1:"y"})

from ggplot import *
ggplot(aes(x="x",y="y"),data=temproute)+geom_path()

# I would like to get route of 10 balloons in one day  store them in a dataframe 
for city in df.iloc[1:,0:2].values:
    start=tuple(df.iloc[0,0:2])
    end=tuple(city)
    

