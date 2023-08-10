import os
import shutil
import glob
import pandas as pd

rerun = ['SRX149192', 'SRX885700', 'ERX402137', 'ERX402138','SRX885698', 'SRX883604','SRX1179573','SRX054533',
'SRX495789', 'SRX1389384','SRX2055961','SRX2055966','SRX2055958', 'ERX402108','SRX330269','ERX402133','SRX306190',
'ERX402112','SRX359797','SRX1433400', 'SRX306193','ERX1403350', 'SRX1179572','SRX1433401','SRX018632','SRX1389387',
 'SRX326970','SRX2055964','SRX885702','SRX2055945','SRX326969', 'SRX447393','SRX330270','SRX495270','SRX2055944',
'SRX097620','SRX359798','SRX883605','SRX018631','SRX306196','SRX018629','SRX2055953','SRX149189','SRX1389388',
'SRX018630','SRX1433397','ERX402114','SRX495269','SRX1433399','SRX495277','SRX495278','SRX495290','SRX495289']

spreadsheet = pd.read_csv('../output/chip/20171103_s2cell_chip-seq.csv')
spreadsheet = spreadsheet[spreadsheet.input != 'no input?']

#Delete everything for the samples I want to rerun
for SRX in rerun:
    srr = spreadsheet[spreadsheet.srx == SRX].srr.values[0].split('|')
    for val in srr:
        SRR = val.strip()
        files = glob.glob('../chipseq-wf/data/chipseq_samples/'+SRR+'/*')
        for f in files:
            try:
                os.remove(f)
            except: 
                shutil.rmtree(f)

# Also removing all of the old symlinked fastqs because I had to move orig files to new location
fastqs = glob.glob('../chipseq-wf/data/chipseq_samples/*/*_R[1,2].fastq.gz')
for file in fastqs:
    os.remove(file)

