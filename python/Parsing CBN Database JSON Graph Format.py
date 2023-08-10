import json
import requests
import time

import networkx as nx

import pybel
from pybel.constants import *
import pybel_tools
from pybel_tools.visualization import to_jupyter

pybel.__version__

pybel_tools.__version__

time.asctime()

res = requests.get("http://causalbionet.com/Networks/GetJSONGraphFile?networkId=hox_2.0_hs").json()

graph = pybel.BELGraph()
parser = pybel.parser.BelParser(graph)

def get_citation(evidence):
    return {
        CITATION_NAME: evidence['citation']['name'],
        CITATION_TYPE: evidence['citation']['type'],
        CITATION_REFERENCE: evidence['citation']['id']
    }

annotation_map = {
    'tissue': 'Tissue',
    'disease': 'Disease',
    'species_common_name': 'Species'
}

species_map = {
    'human': '9606',
    'rat': '10116',
    'mouse': '10090'
}

annotation_value_map = {
    'Species': species_map
}

for edge in res['graph']['edges']:    
    for evidence in edge['metadata']['evidences']:
        if 'citation' not in evidence or not evidence['citation']:
            continue
        
        parser.control_parser.clear()
        parser.control_parser.citation = get_citation(evidence)
        parser.control_parser.evidence = evidence['summary_text'] 
        
        d = {}
        
        if 'biological_context' in evidence:
            annotations = evidence['biological_context']
        
            if annotations['tissue']:
                d['Tissue'] = annotations['tissue']

            if annotations['disease']:
                d['Disease'] = annotations['disease']

            if annotations['species_common_name']:
                d['Species'] = species_map[annotations['species_common_name'].lower()]
        
        parser.control_parser.annotations.update(d)
        bel = '{source} {relation} {target}'.format_map(edge)
        try:
            parser.parseString(bel)
        except Exception as e:
            print(e, bel)

to_jupyter(graph)

