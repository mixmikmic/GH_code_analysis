import sys, os, time
from solc import compile_source, compile_files, link_code
from ethjsonrpc import EthJsonRpc

print("Using environment in "+sys.prefix)
print("Python version "+sys.version)

# Initiate connection to ethereum node
#   Requires a node running with an RPC connection available at port 8545
c = EthJsonRpc('127.0.0.1', 8545)
c.web3_clientVersion()

source = """
pragma solidity ^0.4.2;

contract Example {

    string s="Hello World!";

    function set_s(string new_s) {
        s = new_s;
    }

    function get_s() returns (string) {
        return s;
    }
}"""

# Basic contract compiling process.
#   Requires that the creating account be unlocked.
#   Note that by default, the account will only be unlocked for 5 minutes (300s). 
#   Specify a different duration in the geth personal.unlockAccount('acct','passwd',300) call, or 0 for no limit

compiled = compile_source(source)
# compiled = compile_files(['Solidity/ethjsonrpc_tutorial.sol'])  #Note: Use this to compile from a file
compiledCode = compiled['Example']['bin']
compiledCode = '0x'+compiledCode # This is a hack which makes the system work

# Put the contract in the pool for mining, with a gas reward for processing
contractTx = c.create_contract(c.eth_coinbase(), compiledCode, gas=300000)
print("Contract transaction id is "+contractTx)

print("Waiting for the contract to be mined into the blockchain...")
while c.eth_getTransactionReceipt(contractTx) is None:
        time.sleep(1)

contractAddr = c.get_contract_address(contractTx)
print("Contract address is "+contractAddr)

# There is no cost or delay for reading the state of the blockchain, as this is held on our node
results = c.call(contractAddr, 'get_s()', [], ['string'])
print("The message reads: '"+results[0]+"'")

# Send a transaction to interact with the contract
#   We are targeting the set_s() function, which accepts 1 argument (a string)
#   In the contact definition, this was   function set_s(string new_s){s = new_s;}
params = ['Hello, fair parishioner!']
tx = c.call_with_transaction(c.eth_coinbase(), contractAddr, 'set_s(string)', params)

print("Sending these parameters to the function: "+str(params))
print("Waiting for the state changes to be mined and take effect...")
while c.eth_getTransactionReceipt(tx) is None:
    time.sleep(1)

results = c.call(contractAddr, 'get_s()', [], ['string'])
print("The message now reads '"+results[0]+"'")

