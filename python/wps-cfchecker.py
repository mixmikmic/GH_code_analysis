from owslib.wps import WebProcessingService
token = 'a890731658ac4f1ba93a62598d2f2645'
headers = {'Access-Token': token}
wps = WebProcessingService("https://bovec.dkrz.de/ows/proxy/hummingbird", verify=False, headers=headers)

for process in wps.processes:
    print process.identifier,":", process.title

process = wps.describeprocess(identifier='qa_cfchecker')
for inp in process.dataInputs:
    print inp.identifier, ":", inp.title, ":", inp.dataType

inputs = [('dataset', 'http://bovec.dkrz.de:8090/wpsoutputs/hummingbird/output-b9855b08-42d8-11e6-b10f-abe4891050e3.nc')]
execution = wps.execute(identifier='qa_cfchecker', inputs=inputs, output='output', async=False)
print execution.status

for out in execution.processOutputs:
    print out.title, out.reference

from owslib.wps import ComplexDataInput
import base64
fp = open("/home/pingu/tmp/input2.nc", 'r')
text = fp.read()
fp.close()
encoded = base64.b64encode(text)
content = ComplexDataInput(encoded)
inputs = [ ('dataset', content) ]

execution = wps.execute(identifier='qa_cfchecker', inputs=inputs, output='output', async=False)
print execution.status

for out in execution.processOutputs:
    print out.title, out.reference



