#import all the libraries required

from igraph import *
import igraph.test
igraph.test.run_tests()
print igraph.__version__
import scipy
import numpy
import pandas as pd
import networkx as nx
import scipy.io
from numpy import inf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#IGraph instance
G= igraph.Graph()

#defining node list and edge list
nlist=['Acciaiuoli','Pucci','Pazzi','Salviati','Medici','Barbaderi','Castellani','Peruzzi',
       'Strozzi','Bischeri','Ridolfi','Tornabuoni','Ginori','Albizzi','Guadagni','Lamberteschi']
elist=[('Acciaiuoli','Medici'),('Medici','Salviati'),('Salviati','Pazzi'),('Albizzi','Ginori'),
       ('Peruzzi','Castellani'),('Peruzzi','Strozzi'),('Peruzzi','Bischeri'),('Castellani','Barbaderi'),
       ('Castellani','Strozzi'),('Barbaderi','Medici'),('Medici','Ridolfi'),('Strozzi','Ridolfi'),
       ('Strozzi','Bischeri'),('Ridolfi','Tornabuoni'),('Tornabuoni','Medici'),('Tornabuoni','Guadagni'),
      ('Guadagni','Bischeri'),('Guadagni','Albizzi'),('Albizzi','Medici'),('Guadagni','Lamberteschi')]

#add node list and edge list to the Igraph instance
G.add_vertices(nlist)
G.add_edges(elist)

print(G)

#Calculation of Different types of Centrality measures on G

degreecent= []

# degree centrality
for n in nlist:
    degreecent.append(G.degree(n))

#closeness Centrality    
close=G.closeness(nlist)


# Alternative: closeness centrality
# closeness=[]
# b = G.shortest_paths_dijkstra(source= nlist)
# for bb in b:
#     print bb
#     bb =numpy.array(bb)
#     bb[bb == inf] = 0
#     print bb
#     closeness.append(sum(bb) / float(len(bb)))
#     print sum(bb) / float(len(bb))
#     print("----------")
    
#betweenness centrality
betweenness =G.betweenness(vertices=nlist)

# eigenvector centrality
eigenvec =G.evcent()

#PageRank centrality
pagecen = G.pagerank(vertices=nlist)

nlist

#Dataframe of result
dffinal={"Node": nlist,"Degree Centrality":degreecent,"Closeness Centrality":close,"Betweenness Centrality":betweenness,"Eigenvector Centrality": eigenvec,"Page Rank Centrality":pagecen}
dffinal= pd.DataFrame(dffinal)
print dffinal

#printing result in terms of importance
df1 = dffinal[['Node','Degree Centrality']]
print df1.sort_values(by='Degree Centrality', ascending=0)
print ("-------------------------------------------")
df2 = dffinal[['Node','Closeness Centrality']]
print df2.sort_values(by='Closeness Centrality', ascending=0)
print ("-------------------------------------------")
df3 = dffinal[['Node','Betweenness Centrality']]
print df3.sort_values(by='Betweenness Centrality', ascending=0)
print ("-------------------------------------------")
df4 = dffinal[['Node','Eigenvector Centrality']]
print df4.sort_values(by='Eigenvector Centrality', ascending=0)
print ("-------------------------------------------")
df5 = dffinal[['Node','Page Rank Centrality']]
print df5.sort_values(by='Page Rank Centrality', ascending=0)

#method to convert scipy compressed sparse matrix to igraph without creating overhead of matrix index of zero elements  , it has better efficiency than inbuilt function igraph.Graph.Adjacency()
def scipy_to_igraph(matrix, directed= False):
    sources, targets = matrix.nonzero()    
    return Graph(zip(sources, targets), directed= directed)

import os
#read all the matlab files having compressed sparse matrix
path = "/Users/vsanghvi007/CS591_Network/CS591_hw2"
mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]

# load the .mat file into a dictionary (associative array) d
d = {}

#list to store Modularity Scores
FileName=[]
GenderModularity=[]
SfModularity=[]
MajorModularity=[]
DegreeModularity=[]
Gendersize=[]
Sfsize=[]
Majorsize=[]
Degreesize=[]

for file_name in mat_files:
    scipy.io.loadmat(file_name, d)
    FileName.append(file_name)
    print file_name
    
    id1=[]
    id2=[]
    id3=[]
    gen=[]
    st=[]
    maj=[]
    deg=[]
    
    for keys,values in d.items():
        if keys == "A":
            shape= values.shape
            Gold1=scipy_to_igraph(values)
            Gold2=scipy_to_igraph(values)
            Gold3=scipy_to_igraph(values)
            Gold4=scipy_to_igraph(values)
            degre1=Gold4.degree()
            deg=degre1
            
        if keys=="local_info":
            sf=[]
            gender=[]
            major=[]
            for sub_list in values:
                sf.append(sub_list[0])
                gender.append(sub_list[1])
                major.append(sub_list[2])
            index1= [i for i, e in enumerate(gender) if e == 0]
            index2= [i for i, e in enumerate(sf) if e == 0]
            index3= [i for i, e in enumerate(major) if e == 0]
            gender[:] = [item for item in gender if item != 0]
            sf[:] = [item for item in sf if item != 0]
            major[:] = [item for item in major if item != 0]
#             print gender
#             print index1
#             print sf
#             print index2
#             print major
#             print index3

            id1=index1
            id2=index2
            id3=index3
            gen=gender
            st=sf
            maj=major
            
        
    Gold1.delete_vertices(id1)
    summary(Gold1)
    Gendersize.append(Gold1.vcount())
#     print len(gen)
    GenderModularity.append( Gold1.modularity(gen))
#     print (Gold1.modularity(gen))
   
    Gold2.delete_vertices(id2)
    summary(Gold2)
    Sfsize.append(Gold2.vcount())
#     print len(st)
    SfModularity.append(Gold2.modularity(st))
#     print(Gold2.modularity(st))
    
    Gold3.delete_vertices(id3)
    summary(Gold3)
    Majorsize.append(Gold3.vcount())
#     print len(maj)
    MajorModularity.append(Gold3.modularity(maj))
#     print(Gold3.modularity(maj))    
        
    summary(Gold4)
    Degreesize.append(Gold4.vcount())
#     print len(deg)
    DegreeModularity.append(Gold4.modularity(deg))
#     print(Gold4.modularity(deg))
            
    print("------------------")

    

#Dataframe with all modularity scores and graph size for FB100 Dataset
modular={"File": FileName,"Gender Modularity": GenderModularity,"Student Modularity":SfModularity,"Major Modularity": MajorModularity,"Degree Modularity":DegreeModularity,"Gender Size": Gendersize,"Student Size": Sfsize,"Major Size": Majorsize,"Degree Size": Degreesize}
modular= pd.DataFrame(modular)
print modular

#import required maths libraries
from math import log


X= [log(y,10) for y in modular["Gender Size"]]
Y= modular["Gender Modularity"]
plt.scatter(x=X, y=Y)
plt.xlabel('Network Size')
plt.ylabel('Gender Modularity')
plt.title("Scatter plot of FB100")
#drawing horizontal line at modularity value is 0
plt.axhline(linewidth=2, color='r')
#save the plot
plt.savefig('51.png')

#Histogram /Density plot of Modularity Score for Gender Attribute of Vertex
plt.hist(modular["Gender Modularity"])
plt.xlabel('Gender Modularity')
plt.ylabel('Frequency')
plt.title("Histogram or Density plot of Gender Modularity of FB100")
#drawing vertical line at modularity value is 0
plt.axvline(linewidth=2, color='r')
#save the plot
plt.savefig('55.png')


X= [log(y,10) for y in modular["Student Size"]]
Y= modular["Student Modularity"]
plt.scatter(x=X, y=Y)
plt.xlabel('Network Size')
plt.ylabel('Student Modularity')
plt.title("Scatter plot of FB100")
#drawing horizontal line at modularity value is 0
plt.axhline(linewidth=2, color='r')
#save the plot
plt.savefig('52.png')

#Histogram /Density plot of Modularity Score for Student/Faculty Attribute of Vertex
plt.hist(modular["Student Modularity"])
plt.xlim(-0.05,0.30)
plt.xlabel('Student/Faculty Modularity')
plt.ylabel('Frequency')
plt.title("Histogram or Density plot of Student/Faculty Modularity of FB100")
#drawing vertical line at modularity value is 0
plt.axvline(linewidth=2, color='r')
#save the plot
plt.savefig('56.png')


X=  [log(y,10) for y in modular["Major Size"]]
Y= modular["Major Modularity"]
plt.scatter(x=X, y=Y)
plt.ylim(-0.02, 0.16)
plt.xlabel('Network Size')
plt.ylabel('Major Modularity')
plt.title("Scatter plot of FB100")
#drawing horizontal line at modularity value is 0
plt.axhline(linewidth=2, color='r')
#save the plot
plt.savefig('53.png')

#Histogram /Density plot of Modularity Score for Major Attribute of Vertex
plt.hist(modular["Major Modularity"])
plt.xlabel('Major Modularity')
plt.ylabel('Frequency')
plt.title("Histogram or Density plot of Major Modularity of FB100")
#drawing vertical line at modularity value is 0
plt.axvline(linewidth=2, color='r')
#save the plot
plt.savefig('57.png')


X= [log(y,10) for y in modular["Degree Size"]]
Y= modular["Degree Modularity"]
plt.scatter(x=X, y=Y)
plt.xlabel('Network Size')
plt.ylabel('Degree Modularity')
plt.title("Scatter plot of FB100")
#drawing horizontal line at modularity value is 0
plt.axhline(linewidth=2, color='r')
#save the plot
plt.savefig('54.png')

#Histogram /Density plot of Modularity Score for Degree Attribute of Vertex
plt.hist(modular["Degree Modularity"])
plt.xlabel('Degree Modularity')
plt.ylabel('Frequency')
plt.title("Histogram or Density plot of Degree Modularity of FB100")
#drawing vertical line at modularity value is 0
plt.axvline(linewidth=2, color='r')
#save the plot
plt.savefig('58.png')

