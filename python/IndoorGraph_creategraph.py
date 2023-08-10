import json

graph_dict = ((1,2,10.0), (2,3,3.0))  # (NODE A, NODE B, DISTANCE IN METERS)
with open('test_graph.json', 'w') as f:
    json.dump(graph_dict, f)

