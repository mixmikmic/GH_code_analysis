get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import os, sys
# import_path = os.path.abspath('..')   not necessary ?
install_path = '/home/stephan/Repos/ENES-EUDAT/enes_graph_use_case'
sys.path.append(install_path)
from neo4j_prov import provio

#rov_doc_from_json = provio.get_provdoc('json',install_path+"/neo4j_prov/examples/wps-prov.json")
prov_doc_from_json = provio.get_provdoc('json','/home/stephan/Repos/ENES-EUDAT/submission_forms/test/ingest_prov_1.json')

rels = provio.gen_graph_model(prov_doc_from_json)

print prov_doc_from_json.get_records()
print rels

provio.visualize_prov(prov_doc_from_json)

prov_doc_from_xml = provio.get_provdoc('xml',install_path+"/neo4j_prov/examples/wps-prov.xml")
rels = provio.gen_graph_model(prov_doc_from_xml)

print prov_doc_from_xml.get_records()
print rels

from py2neo import Graph, Node, Relationship, authenticate
authenticate("localhost:7474", "neo4j", "prolog16")

# connect to authenticated graph database
graph = Graph("http://localhost:7474/db/data/")
graph.delete_all()

for rel in rels:
    graph.create(rel)

get_ipython().magic('load_ext cypher')
get_ipython().magic('matplotlib inline')

results = get_ipython().magic('cypher http://neo4j:prolog16@localhost:7474/db/data MATCH (a)-[r]-(b) RETURN a,r, b')
results.get_graph()
results.draw()

from neo4j_prov.vis import draw

options = {"16":"label"}
result_iframe = draw(graph,options)



