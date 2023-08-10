from IPython.display import Image
Image(filename='../Visualization/img/types_of_nodes_and_their_relationships_based_on_mechanisms_and_domains_of_attack_views.png')

from IPython.display import Image
Image(filename='../Visualization/img/capec_website_representation.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/standard_attack_pattern_relationship.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/standard_attack_pattern_relationship_alt.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/meta_attack_pattern_relationship_missing.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/meta_attack_pattern_relationship_missing_alt.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/meta_attack_pattern_relationship.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/meta_attack_pattern_relationship_alt.PNG')

import os
import sys

import csv
import xml.etree.ElementTree as ET

import json

import argparse

parser = argparse.ArgumentParser(description = 'CAPEC XML visualisation program.')
parser.add_argument('xml_filepath', help = 'path of the capec xml file')
parser.add_argument('javascript_filepath', help = 'path of the carrotsearch.foamtree.js file')
parser.add_argument("--json", help = "export .json file", action = "store_true")
parser.add_argument("--gephi", help = "export node and edge table for gephi visualisation", action = "store_true")
args = parser.parse_args()

from IPython.display import Image
Image(filename='../Visualization/img/capec_final_usage.PNG')

filepath = args.xml_filepath
javascript_path = args.javascript_filepath

"""Obtaining the CAPEC xml filename from the path"""
CAPEC_xml = filepath.split(os.sep)[-1]

"""Obtaining the CAPEC version from the filename and using it to rename other files"""
CAPEC_version = CAPEC_xml[:-4]
CAPEC_json = CAPEC_version + ".json"
CAPEC_html = CAPEC_version + ".html"
CAPEC_csv_edgelist = CAPEC_version + " [Edges].csv"
CAPEC_csv_nodelist = CAPEC_version + " [Nodes].csv"

"""This is the inline data model.
   This contains CAPEC_ID, CAPEC_NAME and CHILD_OF nodes.
   This can be extended to have more details.
"""
CAPEC_entries = list()

CAPEC_tree = ET.parse(filepath)
CAPEC_root = CAPEC_tree.getroot()

def remove_commas(string):
    """The purpose of this function is to remove the commas in the string that is passed as input. 
       The need is that some of the CAPEC names contain commas and when eventually added to a CSV file, they occupy multiple 
       cells instead of just one. In order to overcome that we pass the CAPEC name and obtain the "comma-less" version of it.
    """
    comma_free_string = ''
    temp_string = string.split(',')
    for i in temp_string:
        comma_free_string+=i
    return comma_free_string

"""There are three for loops which write to the data model.
   
   The first one is used to write the categories with parent as 1000 - Mechanisms of Attack
   CAPEC_root[0][0] is used to obtain capec:Views and then under it the first tag that contains Mechanisms of attack categories
   
   After obtaining the capec_id, we search for the name in the capec:Categories which is CAPEC_root[1] 
"""
for division in CAPEC_root[0][0]:
    for parameter in division:
        if parameter.tag=="{http://capec.mitre.org/capec-2}Relationship":
            capec_id = parameter[3].text
            child_of = '1000'
            for category in CAPEC_root[1]:
                if category.attrib['ID']==capec_id:
                    capec_name = remove_commas(category.attrib['Name'])
                    temp_list = list()
                    temp_list.extend((capec_id,capec_name,child_of))
                    CAPEC_entries.append(temp_list)

"""The second one is used to write the Meta Attack Patterns onto the data model
   CAPEC_root[1] is used to obtain the capec:Categories tag.
   
   relationship_parameter[2] is capec:Relationship_Nature
   relationship_parameter[3] is capec:Relationship_Target_ID
"""
for attack_pattern in CAPEC_root[1]:
    if attack_pattern.attrib['Status']!="Deprecated":
        for parameter in attack_pattern:
            if parameter.tag=="{http://capec.mitre.org/capec-2}Relationships":
                for relationship_parameter in parameter:
                    if relationship_parameter[2].text=="HasMember":
                        child_of = attack_pattern.attrib['ID']
                        capec_id= relationship_parameter[3].text 
        
                        for attack_pattern_matcher in CAPEC_root[2]:
                            if attack_pattern_matcher.attrib['ID']==capec_id:
                                capec_name= remove_commas(attack_pattern_matcher.attrib['Name'])
                        temp_list = list()
                        temp_list.extend((capec_id,capec_name,child_of))
                        CAPEC_entries.append(temp_list)

"""The third and final loop is used to add the Standard Attack Patterns onto the data model
   CAPEC_root[2] is used to select capec:Attack_Patterns
"""
for attack_pattern in CAPEC_root[2]:
    if attack_pattern.attrib['Status']!="Deprecated":
        for parameter in attack_pattern:
            if parameter.tag=="{http://capec.mitre.org/capec-2}Related_Attack_Patterns":
                for related_attack_pattern_parameter in parameter:
                    if related_attack_pattern_parameter[2].text=="ChildOf":
                        capec_id= attack_pattern.attrib['ID']
                        capec_name= remove_commas(attack_pattern.attrib['Name'])
                        child_of= related_attack_pattern_parameter[3].text
                        temp_list = list()
                        temp_list.extend((capec_id,capec_name,child_of))
                        CAPEC_entries.append(temp_list)

def jsonify(number):
    """This iterative function is used to build the JSON in the format that foamtree javascript requires. 
       It reads off the inline data model which contains the [0]th element as CAPEC_ID, [1] - CAPEC_NAME and 
       [2] - CHILD_OF CAPEC_ID.
       
       The member_dict["label"] is used to name each element in the representation. 
       The current naming scheme is <CAPEC_ID> - <CAPEC_NAME>
    """
    main_list = list()
    for row in CAPEC_entries:
        parent_id = str(number)
        if row[2]==parent_id:
            member_dict = dict()
            member_dict["label"] = str(row[0]) + ' - ' + str(row[1])
            member_dict["weight"] = 1
            member_dict["groups"] = jsonify(row[0])
            main_list.append(member_dict)
        else:
            continue
    return main_list

def gephi_export(number): 
    """This function takes in the View ID that is to be exported. In our program we pass 1000 as it refers to 
       Mechanisms of Attack.
       
       It creates two CSVs - edgelist and nodelist. These can be loaded onto Gephi for visualization.
       
       The edges are created as Directed edges.
    """
    fe = open(CAPEC_csv_edgelist, 'w')
    HEADe = "Source,Target,Type,id,label,timeset,weight\n"
    fe.write(HEADe)
    
    fn = open(CAPEC_csv_nodelist, 'w')
    HEADn = "Id,Label,timeset\n"
    fn.write(HEADn)

    capecid_id_dict = dict()
    capecid_id_dict[str(number)]='0'
    node = '{o1},{o2},{o3}\n'.format(o1='0',o2=number,o3='')
    fn.write(node)
    id_counter = 1
    for row in CAPEC_entries:
        node = '{o1},{o2},{o3}\n'.format(o1=id_counter,o2=row[0],o3='')
        fn.write(node)
        capecid_id_dict[row[0]]=id_counter
        id_counter+=1

    type_value = "Directed"
    edge_id_counter = 0
    for row in CAPEC_entries:
        if row[2] in capecid_id_dict:
            src = int(capecid_id_dict[row[2]])
            tgt = int(capecid_id_dict[row[0]])
            edge = '{o1},{o2},{o3},{o4},{o5},{o6},{o7}\n'.format(o1=src,o2=tgt,o3=type_value,o4=edge_id_counter,o5='',o6='',o7=1)
            fe.write(edge)
            edge_id_counter+=1

def createJSON():
    """This function is used to create the JSON export file."""
    with open(CAPEC_json,'w') as jsonfile:
        json.dump(CAPEC_dict, jsonfile)
    jsonfile.close()

from IPython.display import Image
Image(filename='../Visualization/img/expected_json.PNG')

from IPython.display import Image
Image(filename='sample_json.PNG')

def createHTML():
    """we build the HTML to visualize the JSON using Foamtree javascript.
       foamtreetemplate1 and foamtreetemplate2 are part of the final HTML visualization file.
    """
    foamtreetemplate1 = '''<!DOCTYPE html>
    <html>
      <head>
        <title>FoamTree Quick Start</title>
        <meta charset="utf-8" />
      </head>

      <body>
        <div id="visualization" style="width: 800px; height: 600px"></div>

        <script src="'''+ javascript_path +'''"></script>
        <script>
          window.addEventListener("load", function() {
            var foamtree = new CarrotSearchFoamTree({
              id: "visualization",
              dataObject:'''

    foamtreetemplate2 = '''});
          });
        </script>
      </body>
    </html>'''

    with open(CAPEC_html,'w') as outputfile:
        outputfile.write(foamtreetemplate1)
        for line in open(CAPEC_json):
            outputfile.write(line)
        outputfile.write(foamtreetemplate2)

CAPEC_dict = dict()
CAPEC_dict["groups"]=jsonify(1000)
createJSON()
createHTML()

if args.gephi:
    gephi_export(1000)

if args.json == False:
    os.remove(CAPEC_json)

from IPython.display import Image
Image(filename='../Visualization/img/sample_html.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/capec_v2.9.png')

from IPython.display import Image
Image(filename='../Visualization/img/capec_v1_xmlstructure.PNG')

from IPython.display import Image
Image(filename='capec_v1.7_xmlstructure.PNG')

from IPython.display import Image
Image(filename='../Visualization/img/capec_v2.9_xmlstructure.PNG')

