import sys
sys.path.append("../../qiskit-sdk-py/")
from qiskit import QuantumProgram
import Qconfig
# for visualization
from qiskit.tools.visualization import plot_state

import math, json, copy
import numpy as np

def AddError (script,q,num,simulator,bit):
    
    # errors are rotations around the x axis by a fraction of pi
    # this fraction is twice as large for qubits initially in state 1
    
    fracAncilla = 0.05
    
    fracCode = fracAncilla
    if (bit==1):
        fracCode = fracCode*2

    
    # if the code is simulated add these rotations for error like effects
    if (simulator):
        for address in range(0,num-1,2): # code qubits
            script.u3(fracCode * math.pi, 0.0, 0.0, q[address])
        for address in range(1,num-1,2): # ancilla qubits
            script.u3(fracAncilla * math.pi, 0.0, 0.0, q[address])
        script.u3(fracCode * math.pi, 0.0, 0.0, q[num-1]) # single qubit
                
        script.barrier()

def AddCnot(repetitionScript,q,control,target,simulator):
    
    # set the coupling map ()
    # b in coupling_map[a] means a CNOT with control qubit a and target qubit b can be implemented
    # note that is is not just copy and pasted from https://github.com/IBM/qiskit-qx-info/tree/master/backends/ibmqx3
    coupling_map = {0: [1], 1: [2], 2: [3], 3: [14], 4: [3, 5], 5: [], 6: [7, 11], 7: [10], 8: [7], 9: [10, 8], 10:[], 11: [10], 12: [5, 11, 13], 13: [4, 14], 14:[], 15: [0, 14]}
    
    # if such a CNOT is directly possible, we do it
    if ( target in coupling_map[control] or simulator):
        repetitionScript.cx(q[control], q[target])
    # if it can be done the other way round we conjugate with Hadamards
    elif ( control in coupling_map[target] ):
        repetitionScript.h(q[control])
        repetitionScript.h(q[target])
        repetitionScript.cx(q[target], q[control])
        repetitionScript.h(q[control])
        repetitionScript.h(q[target])
    else:
        print('Qubits ' + str(control) + ' and ' + str(target) + ' cannot be entangled.')

def GetAddress (codeQubit,offset,simulator):
    
    if (simulator):
        address = 2*codeQubit + offset
    else:
        address = (5-2*codeQubit-offset)%16
    
    return address

def RunRepetition(bit,d,device):
    
    # set the number of shots to use on the backend
    shots = 8192
    
    # determine whether a simulator is used
    simulator = (device!='ibmqx3')
    
    # if the simulator is used, we declare the minimum number of qubits required
    if (simulator):
        num = 2*d
    # for the real device there are always 16
    else:
        num = 16
        
    # now to set up the quantum program (QASM)
    Q_program = QuantumProgram()
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url
    # declare register of 5 qubits
    q = Q_program.create_quantum_register("q", num)
    # declare register of 5 classical bits to hold measurement results
    c = Q_program.create_classical_register("c", num)
    # create circuit
    repetitionScript = Q_program.create_circuit("repetitionScript", [q], [c])   
    
    
    # now we insert all the quantum gates to be applied
    # a barrier is inserted between each section of the code to prevent the complilation doing things we don't want it to
    
    # the stored bit is initialized by repeating it accross all code qubits same state
    # since qubits are automatically initialized as 0, we just need to do Xs if b=1
    if (bit==1):
        for codeQubit in range(d):
            repetitionScript.x( q[GetAddress(codeQubit,0,simulator)] )
        # also do it for the single qubit on the end for comparision
        repetitionScript.x( q[GetAddress(d-1,1,simulator)] )
       
    repetitionScript.barrier()
    
    # if the code is simulated add rotations for error like effects (and a barrier)
    AddError(repetitionScript,q,num,simulator,bit)
    
    # we then start the syndrome measurements by doing CNOTs between each code qubit and the next ancilla along the line
    for codeQubit in range(d-1):
        AddCnot(repetitionScript,q,GetAddress(codeQubit,0,simulator),GetAddress(codeQubit,1,simulator),simulator)
    repetitionScript.barrier()
    
    # if the code is simulated add rotations for error like effects (and a barrier)
    AddError(repetitionScript,q,num,simulator,bit)
    
    # next we perform CNOTs between each code qubit and the previous ancilla along the line
    for codeQubit in range(1,d):
        AddCnot(repetitionScript,q,GetAddress(codeQubit,0,simulator),GetAddress(codeQubit,-1,simulator),simulator)
    repetitionScript.barrier()
    
    # if the code is simulated add rotations for error like effects (and a barrier)
    AddError(repetitionScript,q,num,simulator,bit)
    
    # all qubits are then measured
    for address in range(num):
        repetitionScript.measure(q[address], c[address])
        
    # set the APIToken and API url
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
    
    # run the job until actual results are given
    dataNeeded = True
    while dataNeeded:
            
        # compile and run the qasm
        executedJob = Q_program.execute(["repetitionScript"], backend=device, shots = shots, max_credits = 5, wait=5, timeout=600, silent=False)  
        # extract data
        results = executedJob.get_counts("repetitionScript")
        
        # see if it really is data
        if ('status' not in results.keys()):
            dataNeeded = False
        
    
    # the raw data states the number of runs for which each outcome occurred
    # we convert this to fractions before output.
    for key in results.keys():
        results[key] = results[key]/shots
    
    # return the results
    return results

def AddProbToResults(prob,string,results):
    
    if string not in results.keys():
        results[string] = 0
    
    results[string] += prob
    

def CalculateError (encodedBit,results):
    
    # total prob of error will be caculated by looping over all strings
    # we initialize the value to 0
    error = 0
    
    # all strings that have results for the given encoded bit are looped over
    for string in results[encodedBit].keys():

        # the probability P(string|encodedBit) is extracted
        right = results[encodedBit][string]
        
        # as is the probability P(string|!encodedBit)
        # if there is no result for this value in the table, the prob is 0
        wrong = 0
        if string in results[(encodedBit+1)%2].keys():
            wrong = results[(encodedBit+1)%2][string]

        # if this is a string for which P(string|!encodedBit)>P(string|encodedBit), the decoding fails
        # the probabilty P(string|encodedBit) is then added to the error
        if (wrong>right):
            error += right
        # if P(string|!encodedBit)=P(string|encodedBit), the decoder randomly chooses between them
        # P(failure|string) is therefore 0.5 in this case
        elif (wrong==right):
            error += 0.5*right
        # otherwise the decoding succeeds, and we don't care about that
            
    return error

def GetData(device,maxSize,totalRuns):
    
    # loop over code sizes that will fit on the chip (d=3 to d=8)
    for d in range(3,maxSize+1):

        print("**d = " + str(d) + "**")
    
        # do the runs
        for run in range(totalRuns):

            print("**Run " + str(run) + "**")

            # get data for each encoded bit value
            for bit in range(2):

                # run the job and put results in resultsRaw
                resultsRaw = RunRepetition(bit,d,device)

                f = open('Repetition_Code_Results/'+device+'/results_d=' + str(d) + '_run=' + str(run) + '_bit=' + str(bit) + '.txt', 'w')
                f.write( str(resultsRaw) )
                f.close()

def ProcessData(device,encodedBit,maxSize,totalRuns):
    
    # determine whether a simulator is used
    simulator = (device!='ibmqx3')
    
    
    # initialize list used to store the calculated means and variances for results from the codes
    codeResults = [[[0]*4 for _ in range(j)] for j in range(3,maxSize+1)]
    singleResults = [[[0]*2 for _ in range(16)] for _ in range(3,maxSize+1)]
    # singleResults[d-3][j][0] is the probability of state 1 for qubit j when used in a code of distance d
    # singleResults[d-3][j][1] is the variance for the above
    
    
    # the results will show that the case of partial decoding requires more analysis
    # for this reason we will also output combinedCodeResults, which is all runs of codeResults combined
    # here we initialize list of combined results from the code only case
    combinedResultsCode = [[{} for _ in range(3,maxSize+1) ] for _ in range(2)]
    
    
    # loop over code sizes...
    for d in range(3,maxSize+1):    
        # ...and the runs
        for run in range(0,totalRuns):
            
            # we are going to fill a bunch of dictionaries with results
            # each has two copies, one for each possible encoded bit

            # the results that come fresh from the backend
            resultsVeryRaw = [{} for _ in range(2)]
            resultsRaw = [{} for _ in range(2)]
            # the results from the full code (including ancillas)
            # resultsFull[k] gives results for the effective distance d-k code obtained by ignoring the last k code qubits and ancillas
            resultsFull = [[{} for _ in range(d)] for _ in range(2)]
            # the same but with ancilla results excluded
            resultsCode =  [[{} for _ in range(d)] for _ in range(2)]
            # results each single bit
            resultsSingle = [[{} for _ in range(16)] for _ in range(2)]

            # we get results for both possible encoded bits
            for bit in range(2):

                # get results from file
                f = open('Repetition_Code_Results/'+device+'/results_d=' + str(d) + '_run=' + str(run) + '_bit=' + str(bit) + '.txt')
                resultsVeryRaw[bit] = eval(f.read())
                f.close()
                
                # loop over all keys in the raw results and look at the ones without strings as values
                # since all such entries should have a bit string as a key, we call it stringVeryRaw
                for stringVeryRaw in resultsVeryRaw[bit].keys():
                    if resultsVeryRaw[bit][stringVeryRaw] is not str:
                        
                        # create a new dictionary in which each key is padded to a bit string of length 16
                        stringRaw = stringVeryRaw.rjust(16,'0')
                        resultsRaw[bit][stringRaw] = resultsVeryRaw[bit][stringVeryRaw]


                # now stringRaw only has data in the correct format
                # let's loop over its entries and process stuff
                for stringRaw in resultsRaw[bit].keys():

                    # get the prob corresponding to this string
                    probToAdd = resultsRaw[bit][stringRaw]

                    # first we deal with resultsFull and resultsCode

                    # loop over all truncated codes relevant for this d
                    for k in range(d):
                        # distance of this truncated code
                        dd = d-k

                        # extract the bit string relevant for resultsFull
                        # from left to right this will alternate between code and ancilla qubits in increasing order
                        stringFull = ''
                        for codeQubit in range(dd): # add bit value for a code qubit...
                            stringFull += stringRaw[15-GetAddress(codeQubit,0,simulator)]
                            if (codeQubit!=(d-1)): #...and then the ancilla next to it (if we haven't reached the end of the code)
                                stringFull += stringRaw[15-GetAddress(codeQubit,1,simulator)]

                        # remove ancilla bits from this to get the string for resultsCode
                        stringCode = ""
                        for n in range(dd):
                            stringCode += stringFull[2*n]

                        AddProbToResults(probToAdd,stringFull,resultsFull[bit][k])
                        AddProbToResults(probToAdd,stringCode,resultsCode[bit][k])

                    # now we'll do results single

                    # the qubits are listed in the order they are in the code
                    # so for each code qubit
                    for jj in range(8):
                        # loop over it and its neighbour
                        for offset in range(2):
                            stringSingle = stringRaw[15-GetAddress(jj,offset,simulator)]
                            AddProbToResults(probToAdd,stringSingle,resultsSingle[bit][2*jj+offset])

                # combined this run's resultsCode with the total, using the k=0 values
                for stringCode in resultsCode[bit][0].keys():
                    probToAdd = resultsCode[bit][0][stringCode]/10
                    AddProbToResults(probToAdd,stringCode,combinedResultsCode[bit][d-3])
        

            # initialize list used to store the calculated means and variances for results from the codes
            codeSample = [[0]*2 for _ in range(d)]
            # here
            # codeSample gives the results
            # codeSample[0] gives results for the whole code
            # codeSample[k] gives results for the effective distance d-k code obtained by ignoring the last k code qubits and ancillas
            # codeSample[k][0] is the error prob when decoding uses both code and ancilla qubits
            # codeSample[k][1] is the error prob when decoding uses only code qubits
            singleSample = [0]*16
            # singleSample[j] is the probability of state 1 for qubit j when the required bit value is encoded

            # write results in            
            for k in range(d):
                codeSample[k][0] = CalculateError(encodedBit,[resultsFull[0][k],resultsFull[1][k]])
                codeSample[k][1] = CalculateError(encodedBit,[resultsCode[0][k],resultsCode[1][k]])
            for j in range(16):
                if '1' in resultsSingle[encodedBit][j].keys():
                    singleSample[j] = resultsSingle[encodedBit][j]['1']
            
            
            # add results from this run to the overall means and variances
            for k in range(d):
                for l in range(2):
                    codeResults[d-3][k][2*l] += codeSample[k][l] / totalRuns # means
                    codeResults[d-3][k][2*l+1] += codeSample[k][l]**2 / totalRuns # variances
            for j in range(16):
                singleResults[d-3][j][0] += singleSample[j] / totalRuns
                singleResults[d-3][j][1] += singleSample[j]**2 / totalRuns

        # finish the variances by subtracting the square of the mean
        for k in range(d):
            for l in range(1,4,2):
                codeResults[d-3][k][l] -= codeResults[d-3][k][l-1]**2
        for j in range(16):
                singleResults[d-3][j][1] -= singleResults[d-3][j][0]**2

    
    # return processed results                                                                                                                        
    return codeResults, singleResults, combinedResultsCode

# first a quick function to do logs in a nice way
def Log (x):
    if (x>0):
        y = math.log( x , 10 )
    else:
        y = math.nan # the input would cause a domain error, we output a nan    
    return y

def MakeGraph(X,Y,y,axisLabel,labels=[],legendPos='upper right',verbose=False,log=False):
    
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 30})
    
    # if verbose, print the numbers to screen
    if verbose==True:
        print("\nX values")
        print(X)
        for j in range(len(Y)):
            print("\nY values for series "+str(j))
            print(Y[j])
            print("\nError bars")
            print(y[j])
            print("")
    
    # convert the variances of varY into widths of error bars
    for j in range(len(y)):
        for k in range(len(y[j])):
            y[j][k] = math.sqrt(y[j][k]/2)
            if log==True:
                yp = Log(Y[j][k]+y[j][k]) - Log(Y[j][k])
                if (Y[j][k]-y[j][k]>0):
                    ym = Log(Y[j][k]) - Log(Y[j][k]-y[j][k])
                else:
                    ym = 0
                y[j][k] = max(yp,ym)
    
    # if a log plot, do the logs
    if log==True:
        for j in range(len(Y)):
            for k in range(len(Y[j])):
                Y[j][k] = Log(Y[j][k])
    
    
    plt.figure(figsize=(20,10))
    
    
    # add in the series
    for j in range(len(Y)):
        if labels==[]:
            plt.errorbar(X, Y[j], marker = "x", markersize=20, yerr = y[j], linewidth=5)
        else:
            plt.errorbar(X, Y[j], label=labels[j], marker = "x", markersize=20, yerr = y[j], linewidth=5)
    
    plt.legend(loc=legendPos)
    
    # label the axes
    plt.xlabel(axisLabel[0])
    plt.ylabel(axisLabel[1])
    
    # make sure X axis is fully labelled
    plt.xticks(X)

    # make the graph
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    

# set device to use
# this also sets the maximum d. We only go up to 6 on the simulator
userInput = input("Do you want results for the real device? (input Y or N) If not, results will be from a simulator. \n").upper()
if (userInput=="Y"):
    device = 'ibmqx3'
    maxSize = 8
else:
    device = 'local_qasm_simulator'
    maxSize = 6


# determine whether data needs to be taken
userInput = input("Do you have saved data to process (Y/N) If not, new data will be obtained. \n").upper()
if (userInput=="Y"):
    dataAlready = True
else:
    dataAlready = False

# set number of runs used for stats
totalRuns = 10 # should be 10

# if we need data, we get it
if (dataAlready==False):

    # get the required data for the desired number of runs
    GetData(device,maxSize,totalRuns)

codeResults = [[],[]]
singleResults = [[],[]]
for encodedBit in range(2):
    codeResults[encodedBit], singleResults[encodedBit], combinedResultsCode = ProcessData(device,encodedBit,maxSize,totalRuns)

# plot for single qubit data for each code distance
for d in range(3,maxSize+1):
    X = range(16)
    Y = []
    y = []
    # a series for each encoded bit
    for encodedBit in range(2):        
        Y.append([singleResults[encodedBit][d-3][j][0] for j in range(16)])
        y.append([singleResults[encodedBit][d-3][j][1] for j in range(16)])
    # make graph
    print("\n\n***Final state of each qubit for code of distance d = " + str(d) + "***")
    MakeGraph(X,Y,y,['Qubit position in code','Probability of 1'])

for encodedBit in range(2): # separate plots for each encoded bit
    for decoding in ['full','partial']:
        dec = (decoding=='partial') # this is treated as 0 if full and 1 if partial
        X = range(1,maxSize+1)
        Y = []
        y = []
        for d in range(3,maxSize+1):# series for each code size
            seriesY = [math.nan]*(maxSize)
            seriesy = [math.nan]*(maxSize)
            for k in range(d):
                seriesY[d-k-1] = codeResults[encodedBit][d-3][k][2*dec+0]
                seriesy[d-k-1] = codeResults[encodedBit][d-3][k][2*dec+1]
            Y.append(seriesY)
            y.append(seriesy)
        labels = ['d=3','d=4','d=5','d=6','d=7','d=8']
        MakeGraph(X,Y,y,['Effective code distance','Logical error probability'],
                  labels=labels,legendPos = 'upper right')   

for encodedBit in range(2): # separate plots for each encoded bit
    X = range(3,maxSize+1)
    Y = []
    y = []
    for decoding in ['full','partial']:
        dec = (decoding=='partial') # this is treated as 0 if full and 1 if partial
        Y.append([codeResults[encodedBit][d-3][0][2*dec+0] for d in range(3,maxSize+1)])
        y.append([codeResults[encodedBit][d-3][0][2*dec+1] for d in range(3,maxSize+1)])
    MakeGraph(X,Y,y,['Code distance, d','Error probability, P'],
              labels=['Full decoding','Partial decoding'],legendPos='upper right')
     

# for each code distance and each encoded bit value, we'll create a list of the probabilities for each possible number of errors
# list is initialized with zeros
errorNum = [[[0]*(d+1) for d in range(3,maxSize+1)] for _ in range(2)]

for d in range(3,maxSize+1):
        for bit in range(2):
            # for each code distance and each encoded bit value we look at all possible result strings
            for string in combinedResultsCode[bit][d-3]:
                # count the number of errors in each string
                num = 0
                for j in range(d):
                    num += ( int( string[j] , 2 ) + bit )%2
                # add prob to corresponding number of errors
                errorNum[bit][d-3][num] += combinedResultsCode[bit][d-3][string]
        

        # the we make a graph for each, and print a title
        X0 = copy.copy(errorNum[0][d-3]) 
        X1 = copy.copy(errorNum[1][d-3]) # the lists given to MakeGraph can get altered, so we don't put errorNum itself in
        print("\n\n***Probability of errors on code qubits for d = " + str(d) + "***")
        MakeGraph(range(d+1),[X0,X1],[[0]*(d+1)]*2,['Number of code qubit errors','Probability (log base 10)'],
                  labels=['Encoded 0','Encoded 1'],legendPos='upper right',log=True)

        # actually, we make two graphs. This one plots the number of 1s rather than errors, and so the plot for encoded 1 is inverted
        X0 = copy.copy(errorNum[0][d-3]) # X0 in this graph is as before
        X1 = copy.copy(errorNum[1][d-3])[::-1] # but X1 has its order inverted
        print("\n\n***Probability for number of 1s in code qubit result for d = " + str(d) + "***")
        MakeGraph(range(d+1),[X0,X1],[[0]*(d+1)]*2,['Number of 1s in code qubit result','Probability (log base 10)'],
                  labels=['Encoded 0','Encoded 1'],legendPos='center right',log=True)    

get_ipython().run_line_magic('run', '"../version.ipynb"')



