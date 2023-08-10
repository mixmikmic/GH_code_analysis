from py2neo import authenticate, Graph
import py2neo
from neo4j.v1 import GraphDatabase
import json

def read_config():
    data = json.load(open('config.json'))
    # port_name = data["port_name"]
    return data

def connect_to_db():
    params = read_config()
    password = params['password']
    host = params['host']
    http_port = params['http_port']
    user = params['user']
    graph_cnxn = Graph(password=password,host=host,http_port=http_port,user=user)
    return graph_cnxn

graph_cnxn = connect_to_db()

#example query - these are easier to run and view in the browser
# http://ec2-18-218-23-210.us-east-2.compute.amazonaws.com:7474/browser/

graph_cnxn.run("""match p = (n)-[:includes]-(i) where i.name = 'kale' return n,i,p""").dump()

