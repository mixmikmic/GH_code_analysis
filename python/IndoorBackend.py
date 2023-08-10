import websockets
import asyncio

import IndoorGraph as ig
# import IndoorGraph.Error as IGError
from IndoorGraphConfig import *

async def processor(websocket, path):
    nodes = await websocket.recv()
    print("< {}".format(nodes))
    node_source, node_target = nodes.split(':')
    # try:
    graph = ig.create_graph('test_graph.json')
    print(graph)
    path = ig.compute_shortest_path(graph, (node_source, node_target))
    print(path, graph)
    reply = "The shortest path is {}!".format(path)
    # except Exception as e:
    #    reply = str(e)
        
    await websocket.send(reply)
    print("> {}".format(reply))    
    
start_server = websockets.serve(processor, HOST_NAME, PORT)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()



