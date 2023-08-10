from eppy import modeleditor 
from eppy.modeleditor import IDF

IDF.setiddname("Energy+V7_2_0.idd")

wholeidffile = IDF("RefBldgLargeOfficeNew2004_v1.4_7.2_5A_USA_IL_CHICAGO-OHARE.idf")

objectlist = wholeidffile.idfobjects

print objectlist

objectnamelist = []

for objectname in objectlist:
    objectnamelist.append(objectname)

import pandas as pd

nameseries = pd.Series(objectnamelist)

nameseries

nameseries.to_csv('objectnames.csv')

categorizedobjects = pd.read_csv('catagorized_objectnames.csv', header=None, names=['index','objectname','objectcatagory'], index_col='index')

categorizedobjects.objectcatagory.unique()

for category in categorizedobjects.objectcatagory.unique():
    print list(categorizedobjects[(categorizedobjects.objectcatagory == category)].objectname)

wholeidffile.idfobjects["AIRLOOPHVAC"]

objectnamelist = wholeidffile.idfobjects["AIRLOOPHVAC"]

for objectname in objectnamelist:
    print objectname.Name

for category in categorizedobjects.objectcatagory.unique():
    
    #Create a list of the objects in each category
    list_of_category_objs = list(categorizedobjects[(categorizedobjects.objectcatagory == category)].objectname)
    
    #Open the blank IDF file with the same name as the categories
    catIDF = IDF("./blankidftemplates/"+category+".idf")
    
    for catobj in list_of_category_objs:
#         print category + catobj
#         print wholeidffile.idfobjects[catobj]
        
        objectnamelist = wholeidffile.idfobjects[catobj]
        
        if len(objectnamelist) != 0:
#             print objectnamelist
            
            for idfobject in objectnamelist:
                try:
                    print idfobject.Name
                except:
                    print "No Name field"

                #Add each object to the new 'modularized' idf file
                catIDF.copyidfobject(idfobject)
                
    catIDF.saveas("./modularizedidfs/"+category+"_updated.idf")

wholeidffile = IDF("RefBldgLargeOfficeNew2004_v1.4_7.2_5A_USA_IL_CHICAGO-OHARE.idf")

wholeidffile.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']

import numpy
flowpersurfacearea_list = numpy.linspace(0.0001, 0.001, num=10)

flowpersurfacearea_list

for objectinstance in wholeidffile.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']: 
        print objectinstance['Flow_per_Exterior_Surface_Area']

for flowpersurfacearea in flowpersurfacearea_list:
    
    print "Creating IDF file with ZONEINFILTRATION:DESIGNFLOWRATE of "+ str(flowpersurfacearea)

    for objectinstance in wholeidffile.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']:
                
        objectinstance['Flow_per_Exterior_Surface_Area'] = flowpersurfacearea
        
    wholeidffile.saveas("./ParametricIDF/IDF_"+str(flowpersurfacearea)+".idf")
        

