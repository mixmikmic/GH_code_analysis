# Used for the rendering
BLOCK_ID = 364618 #Parasitic Attack
BLOCK_HASH = '00000000000000000a529be68d2cfb3eb80cfd91ce75c4e0aacd388fbe2aa883'

BLOCK_ID = 364281
BLOC = '000000000000000011eb9f6d476f5a4fde5aea89d78898c8d5078733ff1cf098'

import urllib.request
block_info = urllib.request.urlopen("https://blockchain.info/rawblock/" + BLOCK_HASH)

import json
block_info_json = json.load(block_info)
#block_info_json

[a for a in block_info_json]

block_info_json['tx'][0]

#block_info_json['tx'][4]

nodes = []
links = []
for transaction in block_info_json['tx']:
    for input in transaction['inputs']:
        if 'prev_out' in input:
            prev_out = input['prev_out']
            if 'addr' in prev_out:
                address = prev_out['addr']
                if address not in nodes:
                    nodes.append(address)
                for output in transaction['out']:
                    if 'addr' in output:
                        address_out = output['addr']
                        if address_out not in nodes:
                            nodes.append(address_out)  
                        links.append({
                            'source':address,
                            'target':address_out,
                            'value':1
                        })
                    else:
                        print('NO ADDRESS IN ADRESS OUT')
                        print(output)
            else:
                print('ERROR, NO ADDRESS IN PREV OUT')
        else:
            print('NO PREV_OUT')

len(nodes)





nodes_json = [{'id':a,'group':1} for a in nodes]
links_json = links
output = {
    'nodes':nodes_json,
    'links':links
}

import json
with open('data_block.json', 'w') as outfile:
    json.dump(output, outfile)



block_info_json['tx'][1]

nodes = []
nodes_block = []
links = []
for transaction in block_info_json['tx']:
    hash_ = transaction['hash']
    if transaction['hash'] not in nodes_block:
        nodes_block.append(transaction['hash'])
        
    for input in transaction['inputs']:
        if 'prev_out' in input:
            prev_out = input['prev_out']
            if 'addr' in prev_out:
                address = prev_out['addr']
                if address not in nodes:
                    nodes.append(address)
                links.append({
                    'source':address,
                    'target':hash_,
                    'value':1
                })
                    
    for output in transaction['out']:
        if 'addr' in output:
            address_out = output['addr']
            if address_out not in nodes:
                nodes.append(address_out)  
            links.append({
                'source':hash_,
                'target':address_out,
                'value':1
            })

nodes_json = [{'id':a,'group':1} for a in nodes] + [{'id':a,'group':2} for a in nodes_block]
links_json = links
output = {
    'nodes':nodes_json,
    'links':links_json
}

import json
with open('data_block_but_links_are_between_tx_id.json', 'w') as outfile:
    json.dump(output, outfile)



# BLOCK_ID = 
BLOC = '000000000000000012539f9cc04158fd7b52545640c4f74534223ec3812afb0e'

import urllib.request
block_info = urllib.request.urlopen("https://blockchain.info/rawblock/" + BLOCK_HASH)

import json
block_info_json = json.load(block_info)
#block_info_json

nodes = []
nodes_block = []
links = []
for transaction in block_info_json['tx']:
    hash_ = transaction['hash']
    if transaction['hash'] not in nodes_block:
        nodes_block.append(transaction['hash'])
        
    for input in transaction['inputs']:
        if 'prev_out' in input:
            prev_out = input['prev_out']
            if 'addr' in prev_out:
                address = prev_out['addr']
                if address not in nodes:
                    nodes.append(address)
                links.append({
                    'source':address,
                    'target':hash_,
                    'value':1
                })
                    
    for output in transaction['out']:
        if 'addr' in output:
            address_out = output['addr']
            if address_out not in nodes:
                nodes.append(address_out)  
            links.append({
                'source':hash_,
                'target':address_out,
                'value':1
            })

nodes_json = [{'id':a,'group':1} for a in nodes] + [{'id':a,'group':2} for a in nodes_block]
links_json = links
output = {
    'nodes':nodes_json,
    'links':links_json
}

import json
with open('18nov.json', 'w') as outfile:
    json.dump(output, outfile)





