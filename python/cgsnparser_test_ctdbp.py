import sys
import os
import scipy.io as sio
sys.path.insert(0, '/mnt/hgfs/shared_ubuntu/APL/OOI_ipynb/cgsn-parsers/parsers')
from parse_ctdbp import Parser as Parser_ctdbp

pth = "./"
fname = "20160609.ctdbp1.log"
infile = os.path.join(pth,fname)
outfile = os.path.join(pth,'out')

ctdbp = Parser_ctdbp(pth + fname, 1)

print ctdbp.ctd_type
print ctdbp.infile
print ctdbp.data
print ctdbp.load_ascii()
ctdbp.data

ctdbp.parse_data(ctdbp.ctd_type)

ctdbp.data

sio.savemat(outfile, ctdbp.data.toDict())

pth = "./"
fname = "1_20160609.ctdbp1.log"
infile = os.path.join(pth,fname)
outfile = os.path.join(pth,'out1')

ctdbp = Parser_ctdbp(pth + fname, 2)

print ctdbp.ctd_type
print ctdbp.infile
print ctdbp.data
print ctdbp.load_ascii()
ctdbp.data

ctdbp.parse_data(ctdbp.ctd_type)

ctdbp.data

sio.savemat(outfile, ctdbp.data.toDict())



