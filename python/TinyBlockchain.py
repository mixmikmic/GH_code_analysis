import hashlib
import json

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()
        
    def hash_block(self):
        sha = hashlib.sha256()
        sha.update(
            str(self.index) + 
            str(self.timestamp) + 
            str(self.data) + 
            str(self.previous_hash))
        return sha.hexdigest()
    
    def __repr__(self):
        return json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        })
    
    def repr(self):
        return self.__repr__()

import datetime
def create_genesis_block():
    block = Block(
        0, 
        datetime.datetime.now().isoformat(), 
        {
            'proof-of-work': 9,
            'transactions': []
        }, 
        '0'
    )
    return block

def next_block(block):
    return Block(
        block.index + 1, 
        datetime.datetime.now().isoformat(),
        "Hey, I'm block " + str(block.index + 1),
        block.hash
    )

# Create the blockchain and add the genesis block
blockchain = [create_genesis_block()]
previous_block = blockchain[0]

# How many blocks should we add to the chain
# after the genesis block
# num_of_blocks_to_add = 20

# Add blocks to the chain
#for i in range(0, num_of_blocks_to_add):
#    block_to_add = next_block(previous_block)
#    blockchain.append(block_to_add)
#    previous_block = block_to_add
#    # Tell everyone about it!
#    print "Block #{} has been added to the blockchain!".format(block_to_add.index)
#    print "Hash: {}\n".format(block_to_add.hash) 

from flask import Flask
from flask import request
node = Flask(__name__)

transactions = []

@node.route('/transaction', methods=['POST'])
def transaction():
    if request.method == "POST":
        transaction = request.get_json()
        transactions.append(transaction)

        print('New Transaction')
        print(json.dumps(transaction))
        
        return "Transaction submission successful.\n"

import json
miner_address = 'sdiflbasdilcbasdcounawc-random-miner-address-alskcascjnaoscuaocn'

def proof_of_work(last_proof):
    incrementor = last_proof + 1
    while not (incrementor % 9 == 0 and incrementor % last_proof == 0):
        incrementor += 1
        
    return incrementor

@node.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain[-1]
    last_proof = last_block.data['proof-of-work']
    proof = proof_of_work(last_proof)
    transactions.append(
        {"from": "network", "to": miner_address, "amount": 1}
    )
    
    mined_block = Block(
        last_block.index + 1,
        datetime.datetime.now().isoformat(),
        {
            'proof-of-work': proof,
            'transactions': list(transactions)
        },
        last_block.hash
    )
    
    blockchain.append(mined_block)
    
    transactions[:] = []
    
    return repr(mined_block) + '\n'

peer_nodes = []

@node.route('/add_node', methods=['POST'])
def add_node():
    n = request.get_json()
    peer_nodes.append(n['node'])
    
    return "Node added"
    
@node.route('/blocks', methods=['GET'])
def get_blocks():
    consensus()
    
    chain_to_send = blockchain
    
    for block in chain_to_send:
        block = {
            'index': block.index,
            'timestamp': block.timestamp,
            'data': block.data,
            'hash': block.hash
        }
        
        chain_to_send = json.dumps(chain_to_send)
        return chain_to_send
    
def find_new_chains():
    other_chains = []
    for node_url in peer_nodes:
        
        try:
            block = requests.get(node_url + '/blocks').content
        except:
            peer_nodes.remove(node_url)
            
        block = json.loads(block)
        other_chains.append(block)
    return other_chains
    
def consensus():
    other_chains = find_new_chains()
    longest_chain = blockchain
    for chain in other_chains:
        if len(longest_chain) < len(chain):
            longest_chain = chain
    blockchain = longest_chain

node.run(port=16001)



