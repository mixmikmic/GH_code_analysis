import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from cvxpy import *

## Use an off-the-shelf constrained optimization solver (CvxPy)
x = Variable()
z = Variable()

obj = Minimize(x**2 + 2*z**2)
constraints = [x + z == 4]
result = Problem(obj,constraints).solve()

print("Solution at x=%.3f, z=%.3f"%(x.value,z.value))

## Centralized Form

iterations = 20
x,y,z = [[float('NaN')]*iterations for i in range(3)]
x[0],y[0],z[0] = [0,0,0]

rho = 2
c = 4

for i in xrange(1,iterations):
    x[i] = 1/(2.+rho) * (c*rho - y[i-1] - rho*z[i-1])
    z[i] = 1/(4.+rho) * (c*rho - y[i-1] - rho*x[i])
    y[i] = y[i-1] + rho * (x[i] + z[i] - c)
    
print("Solution at x=%.3f, z=%.3f"%(x[-1],z[-1]))
df = pd.DataFrame([x,y,z],index=['x','y','z']).T
df.plot()

from ethjsonrpc import EthJsonRpc, BadResponseError
conn = EthJsonRpc('127.0.0.1', 8545)

x_addr = conn.eth_accounts()[0]
z_addr = conn.eth_accounts()[1]

contractAddr = open('AggSimpleAddress.txt','rb').read()
contractAddr

iterations = 20;
resultDf = pd.DataFrame(index = range(iterations),columns=['x','z','y'])

x_addr = conn.eth_accounts()[0]
z_addr = conn.eth_accounts()[1]

#### ADMM loop: Update x, update z, update y
# General information: this can be called separately by each participant
rho = conn.call(contractAddr,'rho()',[],['int256'])[0]
c   = conn.call(contractAddr,'c()',[],['int256'])[0]

i = 0
resultDf.loc[0,:] = [0,0,0]

for i in range(1, iterations):
    ### X LOCAL CALCULATION: need to retrieve z and y
    z = conn.call(contractAddr,'z()',[],['int256'])[0]
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    x = int( 1/(2.+rho) * (c*rho - y - rho*z) )
    # Submit the x-update
    tx = conn.call_with_transaction(x_addr, contractAddr, 'setX(int256)',[x])
    while conn.eth_getTransactionReceipt(tx) is None:
        time.sleep(0.1) # Wait for the transaction to be mined and state changes to take effect

    # Z LOCAL CALCULATION: This needs to processed after the x-update
    x = conn.call(contractAddr,'x()',[],['int256'])[0]
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    z = int( 1/(4.+rho) * (c*rho - y - rho*x) )
    tx = conn.call_with_transaction(z_addr, contractAddr, 'setZ(int256)',[z])
    while conn.eth_getTransactionReceipt(tx) is None:
        time.sleep(0.2) # Wait for the transaction to be mined and state changes to take effect

    # Update y: this happens on the blockchain
    tx = conn.call_with_transaction(x_addr, contractAddr, 'updateY()',[])
    while conn.eth_getTransactionReceipt(tx) is None:
        time.sleep(0.1) # Wait for the transaction to be mined and state changes to take effect
        
    # Save the values of the state for this iteration:
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    resultDf.loc[i,:] = [x,z,y]

resultDf.plot()

# Use this if you want to reset the variable values to run it again
tx = conn.call_with_transaction(z_addr, contractAddr, 'setX(int256)',[0])
tx = conn.call_with_transaction(z_addr, contractAddr, 'setZ(int256)',[0])
tx = conn.call_with_transaction(z_addr, contractAddr, 'setY(int256)',[0])

x_addr = conn.eth_accounts()[0]
z_addr = conn.eth_accounts()[1]

contractAddr = open('AggregatorAddress.txt','rb').read()
contractAddr

iterations = 20;
resultDf = pd.DataFrame(index = range(iterations),columns=['x','z','y'])

x_addr = conn.eth_accounts()[0]
z_addr = conn.eth_accounts()[1]

updateXfcn = 'submitValue(int256,uint16)'
updateZfcn = 'submitValue(int256,uint16)'
updateYfcn = None

#### ADMM loop: Update x, update z, update y
# General information: this can be called separately by each participant
rho = conn.call(contractAddr,'rho()',[],['int256'])[0]
c   = conn.call(contractAddr,'c()',[],['int256'])[0]

i = 0
resultDf.loc[0,:] = [0,0,0]

for i in range(1, 20):
    ### X LOCAL CALCULATION: need to retrieve z and y
    z = conn.call(contractAddr,'z()',[],['int256'])[0]
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    x = int( 1/(2.+rho) * (c*rho - y - rho*z) )
    # Submit the x-update
    tx = conn.call_with_transaction(x_addr, contractAddr, updateXfcn,[x,i], gas=int(300e3))
    while conn.eth_getTransactionReceipt(tx) is None:
        time.sleep(0.1) # Wait for the transaction to be mined and state changes to take effect

    # Z LOCAL CALCULATION: This needs to processed after the x-update
    x = conn.call(contractAddr,'x()',[],['int256'])[0]
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    z = int( 1/(4.+rho) * (c*rho - y - rho*x) )
    tx = conn.call_with_transaction(z_addr, contractAddr, updateZfcn,[z,i], gas=int(300e3))
    while conn.eth_getTransactionReceipt(tx) is None:
        time.sleep(0.2) # Wait for the transaction to be mined and state changes to take effect

    # Update y: this happens on the blockchain
    if updateYfcn is not None:
        tx = conn.call_with_transaction(x_addr, contractAddr, 'updateY()',[])
        while conn.eth_getTransactionReceipt(tx) is None:
            time.sleep(0.1) # Wait for the transaction to be mined and state changes to take effect
        
    # Save the values of the state for this iteration:
    y = conn.call(contractAddr,'y()',[],['int256'])[0]
    resultDf.loc[i,:] = [x,z,y]
    r = conn.call(contractAddr,'r()',[],['int256'])[0]
    s = conn.call(contractAddr,'s()',[],['int256'])[0]
    solved =  conn.call(contractAddr,'problemSolved()',[],['bool'])[0]
    print("Primal residual r=%s,\t Dual residual s=%s, \t problemSolved is %s"%(r,s,solved))
    if solved: break
    

resultDf.plot()

