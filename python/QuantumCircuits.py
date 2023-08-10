get_ipython().magic('matplotlib inline')
from plot_quantum_circuit import plot_quantum_circuit

class QCircuit:
    def __init__(self):
        self._gates = []
        self._qubits = []
        self._inits = {}
        return
    
    def reset(self): 
        "Just reset the gates, not the qubits"
        self._gates = []
        return
        
    def qubit(self,s,initval=0):
        q = Qubit(s,initval,self)
        self._inits[s] = initval
        self._qubits.append(q)
        return q
    
    def qubits(self,*qs):
        """Initialize qubits given a list of label, initval, label2, ... values
        """
        n = len(qs)/2
        qubits = []
        for i in range(n):
            qubits.append(self.qubit(qs[2*i],qs[2*i+1]))
        return qubits
    
    def zeros(self,ls):
        "Initialize qubits given a list of strings"
        return [self.qubit(l) for l in ls]
    
    def plot(self):
        labels = [q.symbol for q in self._qubits]
        plot_quantum_circuit(self._gates,self._inits,labels)

class Qubit:
    def __init__(self,symbol='', init=None, circuit = None):
        self.symbol = symbol
        self.init = init
        self.circuit = circuit
        return

class Gate:
    def __init__(self,symbol='',unitary=None):
        self.symbol = symbol
        self.unitary = unitary
        return
    
    def __call__(self,*qubits):
        circuit = qubits[0].circuit
        # TODO: put in a check to make sure all qubits are from the same circuit
        circuit._gates.append(tuple([self.symbol]+[qubit.symbol for qubit in qubits]))

# Define some basic gates
X = Gate('X')
Z = Gate('Z')
H = Gate('H')
M = Gate('M')
CNOT = Gate('CNOT')
CX = Gate('CX')
CZ = Gate('CZ')
CR = [Gate(r'$R(2\pi/%d)$' % 2**i) for i in range(20)]

c = QCircuit()
qa,qb = c.zeros('ab')
H(qa)
CNOT(qb,qa)

c.plot()

def plus_minus(q):
    H(q)

c = QCircuit()
qa,qb = c.zeros('ab')
plus_minus(qa)
c.plot()

def share(a,b):
    CNOT(a,b)
share(qb,qa)
c.plot()

c = QCircuit()
qa = c.qubit('q_a',None)
qb = c.qubit('q_b',0)
plus_minus(qa)
share(qb,qa)
c.plot()

def make_bell_pair():
    c = QCircuit()
    a,b = c.zeros('ab')
    H(a)
    CNOT(b,a)
    return a,b

a,b = make_bell_pair()
a.circuit.plot()

def bell(a,b):
    H(a)
    CNOT(b,a)
    return

def alice(q,a):
    CNOT(a,q)
    H(q)
    M(q)
    M(a)
    return q,a

c = QCircuit()
a,b = c.qubits('a',None,'b',None)
alice(a,b)
c.plot()

CZ = Gate('CZ')
CX = Gate('CX')
def bob(b,x,y):
    CX(b,y)
    CZ(b,x)
    return
    

c = QCircuit()
b,x,y = c.qubits('b',0,'x',0,'y',0)
bob(b,x,y)
c.plot()

def teleport(q,a,b):
    bell(a,b)
    x,y = alice(q,a)
    bob(b,x,y)
    return b

c = QCircuit()
q = c.qubit('q',None)
a,b = c.qubits('a',0,'b',0)
teleport(q,a,b)
c.plot()

c = QCircuit()
q,q2,a,a2,b,b2 = c.qubits('q',None,'q2',None,'a',0,'a2',0,'b',0,'b2',0)
teleport(q,a,b)
teleport(q2,a2,b2)
c.plot()

def qft(*qs):
    for i,q in enumerate(qs):
        for j in range(i):
            CR[i-j+1](qs[j],q)
        H(q)
    return qs
        

qc = QCircuit()
a,b,c,d = qc.qubits('a',None,'b',None,'c',None,'d',None)
qft(a,b,c,d)
qc.plot()



