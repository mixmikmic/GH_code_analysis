import json, requests
import pandas as pd

# BASE = "http://localhost:3000/" # Local machine
BASE = "http://192.168.99.100:3000/" # Local Docker instance
# BASE = "http://192.168.99.100:8080/idmapping/v1/" # Agent on Docker
# BASE = "http://52.33.174.107:3000/" # EC2

# BASE = 'http://52.35.61.6:8080/idmapping/v1/'

def jprint(data):
    print(json.dumps(data, indent=4))

# Mixed species query allowed - human, mouse, yeast, and fly
query1 = {
    "ids": ["rAd5", "p53", "mapk1"]
}

res1 = requests.post(BASE + 'map', json=query1)
res_json = res1.json()

jprint(res_json)

# Mixed species query allowed - human, mouse, yeast, and fly
query1 = {
    "ids": ["7157", "P04637"],
    "species": "human"
}

res1 = requests.post(BASE + 'labels', json=query1)
res_json = res1.json()

# print(len(res_json["matched"]))
jprint(res_json)

import json, requests

# Utility function to display JSON
def jprint(data):
    print(json.dumps(data, indent=4))

# Mixed species query allowed - human, mouse, yeast, and fly
query = {
    "ids": ["Antp", "HOXA7"],
    "idTypes": ["GeneID", "Symbol", "UniProtKB-ID", "Synonyms"]
}

res = requests.post(BASE + 'map', json=query1)
jprint(res1.json())

yeast_genes = pd.read_csv("./yeast_genes.txt", names=["GeneID"], dtype={"GeneID": str})
print(len(yeast_genes))

id_list = yeast_genes["GeneID"].tolist()

query_heavy = {
    "ids": id_list, # List of yeast genes
    "species": "yeast"
}

jprint(query_heavy)

q2 = {
    "ids": [
        "YAL003W",
        "YAL030W",
        "YAL038W",
        "YAL040C",
        "YAR007C",
        "YBL005W",
        "YBL021C",
        "YBL026W",
        "YBL050W",
        "YBL069W",
        "YBL079W",
        "YBR018C",
        "YBR019C",
        "YBR020W",
        "YBR043C",
        "YBR045C",
        "YBR050C",
        "YBR072W",
        "YBR093C",
        "YBR109C",
        "YBR112C",
        "YBR118W",
        "YBR135W",
        "YBR155W",
        "YBR160W"],
    "species": "yeast"
}

res_large = requests.post(BASE + 'labels', json=q2)

jprint(res_large.json())

import pandas as pd

large_gene_list = pd.read_csv("./human_genes_list_large.txt", names=["GeneID"], dtype={"GeneID": str})

len(large_gene_list)

id_list = large_gene_list["GeneID"].tolist()

query_heavy = {
    "ids": id_list, # Huge list!
}

res_large = requests.post(BASE + 'map', data=json.dumps(query_heavy), headers=HEADERS)

largeJS = res_large.json()

print(len(largeJS))

# Randomly pick 100 IDs from original list
import random

list_size = len(id_list)

def call_random(server_location):
    random_ids = []

    for i in range(0, 2000):
        next_id = id_list[random.randint(0, list_size-1)]
        random_ids.append(next_id)

    query_rand = {
        "ids": random_ids,
    }

    res_rand = requests.post(server_location + 'map', data=json.dumps(query_rand), headers=HEADERS)

get_ipython().run_cell_magic('timeit', '-n 100', '\ncall_random("http://192.168.99.100:3000/")')

get_ipython().run_cell_magic('timeit', '-n 100', '\ncall_random("http://192.168.99.100:8080/idmapping/v1/")')

