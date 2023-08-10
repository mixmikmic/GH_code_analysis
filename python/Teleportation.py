from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
import numpy as np

quantum_simulator = api.QVMConnection()

p = Program()

# wavefunction to be teleported -- create some random wavefunction
a = np.array([np.random.normal(0, 1) + np.random.normal(0, 1) * 1j for _ in range(4)]).reshape(2, 2)
q, _ = np.linalg.qr(a)
p.defgate("RandomUnitary", q)
p.inst(("RandomUnitary", 2))

wavefunc_orig = quantum_simulator.wavefunction(p)
print ("Original wavefunction: ", wavefunc_orig)

# Create Bell state
p.inst(H(1))
p.inst(CNOT(1, 0))

# Alice measures in the Bell basis
p.inst(CNOT(2, 1))
p.inst(H(2))

alice_regs = [0, 1]

p.measure(1, alice_regs[1]).if_then(alice_regs[1], Program(X(0)), Program(I(0)))
p.measure(2, alice_regs[0]).if_then(alice_regs[0], Program(Z(0)), Program(I(0)))

# teleported wavefunction
wavefunc = quantum_simulator.wavefunction(p)
print ("Teleported wavefunction: ", wavefunc)

