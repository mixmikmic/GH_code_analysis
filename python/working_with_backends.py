# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
from pprint import pprint

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url

pprint(Q_program.available_backends())

pprint(Q_program.get_backend_status('ibmqx2'))

pprint(Q_program.get_backend_configuration('ibmqx2'))

pprint(Q_program.get_backend_configuration('local_qasm_simulator'))

pprint(Q_program.get_backend_calibration('ibmqx2'))

pprint(Q_program.get_backend_parameters('ibmqx2'))

get_ipython().run_line_magic('run', '"../version.ipynb"')



