# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:28:25 2016
Revised 2016-05-17
Revised 2016-05-18 - fixed handling of missing cardinalities in relation source

@author: lhoyle
"""

import requests
import re
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from networkx_viewer import Viewer

# get the most recent xmi.xml file of the PlatformIndependent DDI4 model
#     from the DDI development website
res = requests.get('http://lion.ddialliance.org/xmi.xml')
res.status_code == requests.codes.ok
print('Processing the following version of the DDI4 xmi.xml Platform Independent Model:')
print(res.text[:167])

# parse the root element of the XML file into an XML tree
rootDDI4XMI = ET.fromstring(res.text)
# print(rootDDI4XMI.tag)

# for each view get a list of its classes
#  make a dictionary of those lists keyed on the view name

diagrams = {}

for diagram in rootDDI4XMI.iter('diagram'): 
    view = diagram.find('model').attrib.get('package')
    if view != 'ComplexDataTypes':
        #print(view)
        classList = []
        for element in diagram.find('elements'):
            classList.append(element.attrib.get('subject'))
        #print('   '+str(classList))
        diagrams[view] = classList
#print(str(diagrams))

def viewList(className, diagrams ):
# given a class name return a list of the Views in which it appears     
    viewList = []
    for viewName, viewClassList in diagrams.items():        
        if className in viewClassList:
            viewList.append(viewName)
    if len(viewList)>0:
        return viewList
    else:
        return        
def cardinalityString(lowerValue, upperValue):
# Given the lower and upper values from an xmi file, return one string e.g. "0..n"
    if lowerValue == None:
        lowerValue = "0"
    elif lowerValue == "-1":
        lowerValue = "n"
    if upperValue == None:
        upperValue = "0"
    elif upperValue == "-1":
        upperValue = "n" 
    return lowerValue + ".." + upperValue  

# create an overall model graph
overallGraph = nx.DiGraph()

# this dictionary will contain a list of properties for each class with properties
classProperties = {}

# this dictionary will contain a list of relations for each class with relations
classRelations = {}

# this dictionary will contain true (is abstract) or false (is not abstract)  for all classes 
classIsAbstract = {}

#this dictionary will contain the parent class name for any class that extends its parent
classParentName = {}

# this dictionary will contain the target cardinality keyed on the relation (association) name
associationTargetCardinality = {}



# populate the classIsAbstract dictionary , create a node in the graph
for packagedElement in rootDDI4XMI.iter('packagedElement'):
    # only want xmi:type="uml:Class"
    if packagedElement.attrib.get('{http://www.omg.org/spec/XMI/20110701}type') == "uml:Class":
        name = packagedElement.attrib.get('name')
        if packagedElement.attrib.get('isAbstract') == 'true':
            classIsAbstract[name] = True
        else:
            classIsAbstract[name] = False
        overallGraph.add_node(name,{'isAbstract':classIsAbstract[name]} )     
    
# populate classParentName with the classes that extend another
for generalization in rootDDI4XMI.iter('generalization'):
    xmiId = generalization.attrib.get('{http://www.omg.org/spec/XMI/20110701}id')
    xmiIdRegex = re.compile(r'''(
           ^([^_]+)_   # child class
           ([^_]+)_    # "extends"
           ([^_]*)$    # parent class
           )''', re.VERBOSE)
    xmiIdSearch = xmiIdRegex.search(xmiId)
    child=xmiIdSearch.group(2)
    parent=xmiIdSearch.group(4)
    classParentName[child] = parent
    overallGraph.add_node(child,{'isAbstract':classIsAbstract[name], 'extends':parent} )
    
    
#populate tne associationTargetCardinality dictionary
for ownedEnd in rootDDI4XMI.iter('ownedEnd'):
    association = ownedEnd.attrib.get('association')
    
    if ownedEnd.find('lowerValue') == None or ownedEnd.find('upperValue') == None:
        targetCardinality = "Missing"
        print("NOTE: missing target cardinality for " + association)
    else:    
        targetCardinality = cardinalityString(ownedEnd.find('lowerValue').attrib.get('value'),ownedEnd.find('upperValue').attrib.get('value'))
    
        
    associationTargetCardinality[association] = targetCardinality
    
# find the Source, Target and relationship name for each relationship in the DDI4 model

for ownedAttribute in rootDDI4XMI.iter('ownedAttribute'):
    association = ownedAttribute.attrib.get('association')
    # NOTE: assumption - Relations all have an association attribute
    if association !=  None:
        associationRegex = re.compile(r'''(
           ^([^_]+)_   # source class
           ([^_]+)_    # association name
           ([^_]*)$    # "association"
           )''', re.VERBOSE)
        oASearch = associationRegex.search(association)
        relationName = oASearch.group(3)
        relationSource = ownedAttribute.attrib.get('name')
        
        relationTarget = ownedAttribute.find('type').attrib.get('{http://www.omg.org/spec/XMI/20110701}idref')
        
        #  extract and edit cardinalities into one string
        
        if ownedAttribute.find('lowerValue') == None:
            lowerCardinalityValue = "Missing"
            print("NOTE: for" + relationSource + "relation " + relationName + "cardinality is missing")                 
        else:
            lowerCardinalityValue = ownedAttribute.find('lowerValue').attrib.get('value')
        
        if ownedAttribute.find('upperValue') == None:
            upperCardinalityValue = "Missing"
            print("NOTE: for" + relationSource + "relation " + relationName + "cardinality is missing")                 
        else:
            upperCardinalityValue = ownedAttribute.find('upperValue').attrib.get('value')   
            
        sourceCardinality = cardinalityString(lowerCardinalityValue, upperCardinalityValue)
        targetCardinality = associationTargetCardinality[association]
        relationCardinality = sourceCardinality + "->" + targetCardinality

        # print(relationSource, relationName, relationTarget)
        
        # put relationNames in a list for each class
        if relationSource in classRelations.keys():
            # add a relation to this class's list
            classRelations[relationSource].append(relationName)
        else:
            classRelations[relationSource] = [relationName]
            
        # add nodes and edge to the graph
        overallGraph.add_node(relationSource,{'properties':classProperties.get(relationSource), 'relations':classRelations.get(relationSource), 'inViews':viewList(relationSource,diagrams),'isAbstract':classIsAbstract[relationSource], 'extends':classParentName.get(relationSource)} )    
        
        #print(relationSource, relationName, relationCardinality)
        overallGraph.add_edge(relationSource, relationTarget, name=relationName, cardinality=relationCardinality)
    else:
        # put properties into a list for each class in a dictionary
        xmiId = ownedAttribute.attrib.get('{http://www.omg.org/spec/XMI/20110701}id')
        idRegex = re.compile(r'''(
             ^([^_]+)_   # class name
             ([^_]*)$    # property
        )''', re.VERBOSE)
        idSearch = idRegex.search(xmiId)
        className = idSearch.group(2)
        propertyName = idSearch.group(3)
        #  extract and edit cardinalities into one string
        propertyCardinality = cardinalityString(ownedAttribute.find('lowerValue').attrib.get('value'),ownedAttribute.find('upperValue').attrib.get('value'))     
        
        if className in classProperties.keys():
            # add a property to this class's list
            classProperties[className].append(propertyName+"("+propertyCardinality+")")
        else:
            classProperties[className] = [propertyName+"("+propertyCardinality+")"]
        overallGraph.add_node(className,{'properties':classProperties.get(className), 'relations':classRelations.get(className), 'inViews':viewList(className,diagrams),'isAbstract':classIsAbstract[className], 'extends':classParentName.get(className)} )    



posSpring =  nx.spring_layout(overallGraph) 
posShell = nx.shell_layout(overallGraph) 

# Reports and Graphs for each View
for viewName, classList in diagrams.items():
    print("\n\nView " + viewName)
    viewPropertyList = []
    for className in classList:
        classPropertyList = classProperties.get(className)
        if classPropertyList != None:
            for propertyName in classPropertyList:
                viewPropertyList.append(propertyName + "(" + className + ")"  )
    viewPropertyList.sort()
    print("\nProperty List")
    for propertyName in viewPropertyList:
        print(propertyName)
    
    # draw a static diagram of each view both as spring-form and circular
    viewGraph = overallGraph.subgraph(classList).copy()
    #print("View: "+viewName)
    print("\nNodes: " + str(viewGraph.nodes()))
    print("\nEdges: " + str(viewGraph.edges()))
    graphLabel = viewName + "_Springform_Layout"
    nx.draw_spring(viewGraph, with_labels=True, node_size=600, node_color='#eeffff', font_size=9) 
    plt.suptitle(graphLabel)
    plt.savefig(viewName + "_spring.png")
    plt.show()
    graphLabel = viewName + "_Circular_Layout"
    nx.draw_circular(viewGraph, with_labels=True, node_size=600, node_color='#eeffff', font_size=9) 
    plt.suptitle(graphLabel)
    plt.savefig(viewName + "_circular.png")
    plt.show()

  
# use the interactive viewer to visualize the graph of the whole model     
app = Viewer(overallGraph)
app.mainloop()    
    

    
    
    

    


    





