import urllib, requests, json, time
bb_url = 'https://translator.ncats.io/blackboard/api/kg'
#bb_url = 'http://localhost:9000/blackboard/api/kg'

#query = {"type": "query", "name": "A simple blackboard example","term": "asthma"}
query = {"type":"drug", "uri":"https://pharos.ncats.io/idg/api/v1/ligands(31242)", 
         "name":"imatinib", "description": "This is an example query seeded with gleevec"}
json.dumps(query)
req = requests.post(bb_url, json=query)
req.status_code

kg = requests.get(bb_url).json()
latest_kg_id = max([x['id'] for x in kg])
kg_url = bb_url+"/"+str(latest_kg_id)
print('Currently have %d KGs. Will use KG ID=%d, %s' % (len(kg), latest_kg_id, kg_url))

req = requests.put(kg_url+"/ks.pharos")
req.status_code
time.sleep(10)

req = requests.get(kg_url+"?view=full")
req.json()

req = requests.put(kg_url+"/ks.pharos")
time.sleep(30)

## Dump out the KG JSON for inspection
req = requests.get(kg_url+"?view=full")
ofile = open('kg.json', 'w') ## for reference
ofile.write(req.text)
ofile.close()

from igraph import *

def kg2ig(kg):
    if kg['type'] != 'kgraph':
        raise Exception("Must provide a JSON kgraph")

    g = Graph(directed=False)

    nodes = kg['nodes']
    for node in nodes:
        d = {}
        for key in node.keys():
            if key in ['inDegree','outDegree','degree', 'id']: continue
            key = key.encode("ascii")
            d[key] = node[key]
        g.add_vertex(**d)

    edges = kg['edges']
    for edge in edges:
        s = list(filter(lambda x: x['id'] == edge['source'], nodes))
        t = list(filter(lambda x: x['id'] == edge['target'], nodes))
        if len(s) == 1 and len(t) == 1:
            g.add_edge(s[0]['name'],t[0]['name'], type=edge['type'])
        
    return g

req = requests.get(kg_url+"?view=full")
g = kg2ig(json.loads(req.text))
print(g.summary())

from cyjs import *
cy = cyjs()
display(cy)

cy.deleteGraph()
cy.addGraph(g)
kkLayout = g.layout("kk")
cy.setPosition(kkLayout)
cy.fit(10)

cy.loadStyleFile('bb-style.js')

from ndex.networkn import NdexGraph
import ndex.client as nc
import uuid
ng = NdexGraph()
ng.set_name(kg_url)
for n in g.vs:
    ng.add_new_node(n['name'], n.attributes())
for idx, e in enumerate(g.es):
    eid = ng.add_edge_between(e.source+1, e.target+1)
    ng.set_edge_attribute(eid, 'type', e['type'])

client = nc.Ndex("http://dev.ndexbio.org", 'foobar123', 'hello123')
uri = client.save_new_network(ng.to_cx())
uuid = uri.rpartition('/')[-1]
##client.make_network_public(uuid)
print("KG is on Ndex at %s" % (uri))



