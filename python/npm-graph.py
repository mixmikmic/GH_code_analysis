from py2neo import Graph
import json
import requests

REGISTRY_URL = "https://skimdb.npmjs.com/registry/"

graph = Graph()

def loadPackage(package_name):
    r = requests.get(REGISTRY_URL + package_name)
    d = r.json()
    latest_tag = d['dist-tags']['latest']
    #current_version = d['versions'][latest_tag]
    print(latest_tag)
    graph.cypher.execute(PACKAGE_IMPORT_QUERY, parameters={'json': d, 'current_version': latest_tag})
    #return d, current_version

# TODO: create (Package)-[:LATEST]->(Version)
# TODO: rename Package to Module
PACKAGE_IMPORT_QUERY = '''
WITH {json} AS data 
MERGE (p:Package {package_id: data._id})
SET p._rev = data._rev,
    p.name = data.name,
    p.description = data.description

MERGE (author:Developer {email: coalesce(data.author.email, "N/A")})
SET author.name = data.author.name
CREATE UNIQUE (p)<-[:AUTHOR_OF]-(author)

MERGE (rep:Repository {url: coalesce(data.repository.url, "N/A")})
SET rep.type = data.repository.type
CREATE UNIQUE (p)-[:HAS_REPO]->(rep)

FOREACH (maint IN data.maintainers | 
    MERGE (m:Developer {email: coalesce(maint.email, "N/A")})
    SET m.name = maint.name
    CREATE UNIQUE (m)-[:MAINTAINS]->(p)
)

FOREACH (cont IN data.contributors |
    MERGE (c:Developer {email: coalesce(cont.email, "N/A")})
    SET c.name = cont.name
    CREATE UNIQUE (c)-[:CONTRIBUTES_TO]->(p)
)

FOREACH (kw IN data.keywords |
    MERGE (k:Keyword {word: kw})
    CREATE UNIQUE (k)<-[:DEALS_WITH]-(p)
)


WITH data, p
UNWIND keys(data.versions) AS cv

MERGE (v:Version {version: data.versions[cv]["version"]})<-[:HAS_VERSION]-(p)


MERGE (l:License {name: coalesce(data.versions[cv]["license"], "N/A")})
CREATE UNIQUE (v)-[:LICENSED_UNDER]->(l)

FOREACH (dep IN keys(data.versions[cv]["dependencies"]) |
    MERGE (dep_p:Package {package_id: dep})
    MERGE (dep_v:Version {version: replace(replace(data.versions[cv]["dependencies"][dep], "~", ""), ">=", "")})<-[:HAS_VERSION]-(dep_p)
    //MERGE (dep_v)<-[:HAS_VERSION]-(dep_p)
    //SET dep_p.version = replace(replace(current_version.dependencies[dep], "~", ""), ">=", "")
    CREATE UNIQUE (dep_v)<-[:DEPENDS_ON]-(v)
)




'''

GET_PENDING_PACKAGE_QUERY = '''
MATCH (p:Package) WHERE NOT has(p.name) AND NOT has(p.crawled)
WITH p LIMIT 1
SET p.crawled = true
RETURN p.package_id AS package_id LIMIT 1
'''

# start with a single module and crawl from there
loadPackage("express")

graph.cypher.execute(PACKAGE_IMPORT_QUERY, parameters={'json': package_json, 'current_version': latest_tag})

def crawlRegistry():
    result = graph.cypher.execute(GET_PENDING_PACKAGE_QUERY)
    while result:
        new_package = result.one
        print(new_package)
        try:
            loadPackage(new_package)
            result = graph.cypher.execute(GET_PENDING_PACKAGE_QUERY)
        except:
            crawlRegistry()

crawlRegistry()





