from web3 import Web3, HTTPProvider, IPCProvider

gethRPCUrl='http://192.168.1.185:8545'
web3 = Web3(HTTPProvider(gethRPCUrl))

# Retrieve the last block number available from geth RPC
currentblock = web3.eth.getBlock('latest').number
print("Latest block: " + str(currentblock))

web3.eth.getBlock(1,full_transactions=True)

dict(web3.eth.getBlock(46147,full_transactions=True))

web3.fromWei(31337, unit='ether')

def getBlockRange(blockstart,blockend):
    blocksDict = [ ]
    
    for block in range(blockstart,blockend):
        
        blocksDict.append(dict(web3.eth.getBlock(block,full_transactions=False)))
        
    return blocksDict

threeblocks=getBlockRange(46147,46150)
len(threeblocks)

threeblocks[0].keys()

threeblocks[0]

def getTransactionsInRange(blockstart,blockend):
    transactions_in_range=[]
    for block in range(blockstart,blockend):       
        transactions_in_block = web3.eth.getBlock(block,full_transactions=True)['transactions']
        
        # Append as dict(transaction)'s  '{}', to remove "AttributeDict" from each entry
        for transaction in transactions_in_block:
            transactions_in_range.append(dict(transaction))
    return transactions_in_range    

currentblock = web3.eth.getBlock('latest').number
txs_view = getTransactionsInRange(blockstart=currentblock-3,
                                  blockend=currentblock)
print(str(len(txs_view)) + " Transactions in current block\n")


print(txs_view[0])

txs_view[0].keys()

web3.txpool.inspect()

get_ipython().system('pip show web3')

web3.txpool.content



