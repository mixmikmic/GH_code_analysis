get_ipython().system('curl https://raw.githubusercontent.com/Serulab/Py4Bio/master/samples/samples.tar.bz2 -o samples.tar.bz2')
get_ipython().system('mkdir samples')
get_ipython().system('tar xvfj samples.tar.bz2 -C samples')

from Bio.SeqUtils import MeltingTemp as MT

PRIMER_FILE = 'samples/primers.txt'
for line in open(PRIMER_FILE):
    # prm stores the primer, without 5'- and -3'
    prm = line[3:len(line)-4].replace(' ','')
    # .2f is used to print up to 2 decimals.
    print('{0},{1:.2f}'.format(prm, MT.Tm_staluc(prm)))

from Bio.SeqUtils import MeltingTemp as MT
import xlwt

PRIMER_FILE = 'samples/primers.txt'
# w is the name of a newly created workbook.
w = xlwt.Workbook()
# ws is the name of a new sheet in this workbook.
ws = w.add_sheet('Result')
# These two lines writes the titles of the columns.
ws.write(0, 0, 'Primer Sequence')
ws.write(0, 1, 'Tm')
for index, line in enumerate(open(PRIMER_FILE)):
    # For each line in the input file, write the primer
    # sequence and the Tm
    prm = line[3:len(line)-4].replace(' ','')
    ws.write(index+1, 0, prm)
    ws.write(index+1, 1, '{0:.2f}'.format(MT.Tm_staluc(prm)))
# Save the spreadsheel into a file.
w.save('primerout.xls')

