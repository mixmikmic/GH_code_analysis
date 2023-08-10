import pandas as pd
import glob
import os
import ruamel.yaml as YAML
import string

spreadsheet = pd.read_csv('../output/chip/20171103_s2cell_chip-seq.csv')
spreadsheet.head()

#For now we are excluding datasets with no input: 
spreadsheet = spreadsheet[spreadsheet.input != 'no input?']

all_chromatin = spreadsheet[spreadsheet.chromatin == 1]
no_chromatin = spreadsheet[spreadsheet.chromatin == 0]
chrom_srr_list = all_chromatin.srr.values.tolist()
tf_srr_list = no_chromatin.srr.values.tolist()

#New fill in for new data location to run after all data is copied
SRR_list = spreadsheet.srr.values.tolist() 
table = []
missing_files = []

def sanitize_fname(fname):
        valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
        return ''.join([x for x in fname if x in valid_chars])

for val in SRR_list:
    for srr in val.split('|'):
        SRR = srr.strip()
        row = spreadsheet[spreadsheet.srr.astype(str).str.contains(SRR)]
        antibody = sanitize_fname(row.target.values[0])
        srx = row.srx.values[0]
        inpt = row.input.values[0]
        biomat = 's2cell-'+inpt
        PATH = '/data/Oliverlab/data/SRA_s2_chip/'
        myglob = glob.glob(PATH+SRR+'_*')
        if myglob: 
            filename = myglob[0]
            new_row = [SRR, antibody, biomat, '1', srx, filename]
            table.append(new_row)
        else:
            missing_files.append(srr) 
                                   
        if inpt != 'no input?':
            inpt_row = spreadsheet[spreadsheet.srx.astype(str).str.contains(inpt)]
            inpt_srr = inpt_row.srr.values[0]
            if inpt_srr not in SRR_list: 
                SRR_list.append(inpt_srr)
    
my_sampletable = pd.DataFrame(table, columns=['samplename','antibody','biological_material','replicate','label',
                                             'orig_filename']) 
#write out 
my_sampletable.to_csv('../chipseq-wf/config/sampletable_all.tsv', sep='\t', index=False)

missing_files

my_sampletable.head()

spp_empty = ['SRX149192', 'SRX885700', 'ERX402137', 'ERX402138','SRX885698', 'SRX883604','SRX1179573','SRX054533',
'SRX495789', 'SRX1389384','SRX2055961','SRX2055966','SRX2055958', 'ERX402108','SRX330269','ERX402133','SRX306190',
'ERX402112','SRX359797','SRX1433400', 'SRX306193','ERX1403350', 'SRX1179572','SRX1433401','SRX018632','SRX1389387',
 'SRX326970','SRX2055964','SRX885702','SRX2055945','SRX326969', 'SRX447393','SRX330270','SRX495270','SRX2055944',
'SRX097620','SRX359798','SRX883605','SRX018631','SRX306196','SRX018629','SRX2055953','SRX149189','SRX1389388',
'SRX018630','SRX1433397','ERX402114','SRX495269','SRX1433399']

macs_empty = ['SRX495277','SRX495278','SRX495290','SRX495289']

#pull out labels from sample table that are not inputs  
#chromatin extra
#extra = '-g dm --bdg --broad --slocal 5000 --nomodel --extsize 147'
#tf extra
extra = '-g dm --bdg --nomodel --extsize 147'
SRX_list = set(my_sampletable.label.values)

block_list = []
with open('../chipseq-wf/config/config_tf.yaml', 'w') as outfile: 
    for SRX in SRX_list: 
        row = spreadsheet[spreadsheet.srx.astype(str).str.contains(SRX)]
        if not row.antibody.astype(str).str.contains('input').bool():
            label = SRX
            ip = SRX
            control = row.input.values[0]
            macs_block = {'label': label, 'algorithm': 'macs2', 'ip': [ip], 'control': [control], 
                                    'extra': YAML.scalarstring.SingleQuotedScalarString(extra)}
            spp_block = {'label': label, 'algorithm': 'spp', 'ip': [ip], 'control': [control], 
                                    'extra': {'fdr': 0.001}}
            if macs_block and spp_block not in block_list: 
                block_list.append(macs_block)
                block_list.append(spp_block)
    with open('../chipseq-wf/config/config.yaml', 'r') as c:
        page = YAML.load(c, Loader=YAML.RoundTripLoader, preserve_quotes=True)
        for block in block_list: 
            page['chipseq']['peak_calling'].append(block)
        YAML.dump(page, outfile, Dumper=YAML.RoundTripDumper)

#fill in config with samples I want to rerun: 
my_srrs = spreadsheet[spreadsheet.srx.isin(spp_empty)].srr.values.tolist()

extra = '-g dm --bdg --nomodel --extsize 589'
SRX_list = set(spp_empty)
macs_list = set(macs_empty)

block_list = []
with open('../chipseq-wf/config/config_empty.yaml', 'w') as outfile: 
    for SRX in SRX_list: 
        row = spreadsheet[spreadsheet.srx.astype(str).str.contains(SRX)]
        if not row.antibody.astype(str).str.contains('input').bool():
            label = SRX
            ip = SRX
            control = row.input.values[0]
            spp_block = {'label': label, 'algorithm': 'spp', 'ip': [ip], 'control': [control], 
                                    'extra': {'fdr': 0.1}}
            if spp_block not in block_list: 
                block_list.append(spp_block)
    for SRX in macs_list: 
        row = spreadsheet[spreadsheet.srx.astype(str).str.contains(SRX)]
        if not row.antibody.astype(str).str.contains('input').bool():
            label = SRX
            ip = SRX
            control = row.input.values[0]
            macs_block = {'label': label, 'algorithm': 'macs2', 'ip': [ip], 'control': [control], 
                                    'extra': YAML.scalarstring.SingleQuotedScalarString(extra)}
            if macs_block not in block_list: 
                block_list.append(macs_block)
    with open('../chipseq-wf/config/config.yaml', 'r') as c:
        page = YAML.load(c, Loader=YAML.RoundTripLoader, preserve_quotes=True)
        for block in block_list: 
            page['chipseq']['peak_calling'].append(block)
        YAML.dump(page, outfile, Dumper=YAML.RoundTripDumper)

#Moving out of the way for now because I don't think I need anymore
#ALSO move everything to new data folder: /data/Oliverlab/data/SRA_s2_chip

#SRR_list = spreadsheet.srr.values.tolist() 
#table = []
#missing_files = []

#def sanitize_fname(fname):
#        valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
#        return ''.join([x for x in fname if x in valid_chars])
#
#for val in SRR_list:
#    for srr in val.split('|'):
#        SRR = srr.strip()
#        row = spreadsheet[spreadsheet.srr.astype(str).str.contains(SRR)]
#        antibody = sanitize_fname(row.target.values[0])
#        srx = row.srx.values[0]
#        inpt = row.input.values[0]
#        biomat = 's2cell-'+srx
#        PATH = '/data/MiegNCBI/ncbi_remap/prealn-wf/output/samples/'+srx+'/'+SRR
#        with open(glob.glob(PATH+'/LAYOUT')[0]) as f:
#            for line in f:
#                if os.path.isfile(PATH+'/'+SRR+'_1.fastq.gz'):
#                    if [line == 'SE'] and [os.path.getsize(PATH+'/'+SRR+'_1.fastq.gz') > 0]: 
#                        filename = PATH+'/'+SRR+'_1.fastq.gz'
#                    if line == 'SE' and os.path.getsize(PATH+'/'+SRR+'_2.fastq.gz') > 0:
#                        filename = PATH+'/'+SRR+'_2.fastq.gz'
#                    if line == 'keep_R1':
#                        filename = PATH+'/'+SRR+'_1.fastq.gz'
#                    if line == 'keep_R2':
#                        filename = PATH+'/'+SRR+'_2.fastq.gz'
#                    if line == 'PE': 
#                        filename = PATH+'/'+SRR+'_1.fastq.gz'
#                        #decided to get rid of the 2nd
#                        #row_2 = [SRR+'_2', antibody, 's2cell', '2', srx, PATH+'/'+SRR+'_2.fastq.gz']
#                    new_row = [SRR, antibody, biomat, '1', srx, filename]
#                    table.append(new_row)
#                else:
#                    myglob = glob.glob('/home/bergeric/data/s2cell-prior/output/chip/fastq_dump/'+SRR+'_*')
#                    if myglob: 
#                        filename = myglob[0]
#                        new_row = [SRR, antibody, biomat, '1', srx, filename]
#                        table.append(new_row)
#                    else:
#                        filename = 'missing'
#                        new_row = [SRR, antibody, biomat, '1', srx, filename]
#                        table.append(new_row)
#                                   
#        if inpt != 'no input?':
#            inpt_row = spreadsheet[spreadsheet.srx.astype(str).str.contains(inpt)]
#            inpt_srr = inpt_row.srr.values[0]
#            if inpt_srr not in SRR_list: 
#                SRR_list.append(inpt_srr)
#    
#my_sampletable = pd.DataFrame(table, columns=['samplename','antibody','biological_material','replicate','label',
#                                             'orig_filename']) 
#paired end only: 
#paired_end = my_sampletable[my_sampletable.samplename.astype(str).str.contains('_')]
#single end only:
#single_end = my_sampletable[~my_sampletable.samplename.astype(str).str.contains('_')]
#write out 
#my_sampletable.to_csv('/data/Oliverlab/data/SRA_s2_chip/sampletable.tsv', sep='\t', index=False)
#my_sampletable.to_csv('../chipseq-wf/config/sampletable_all.tsv', sep='\t', index=False)
#single_end.to_csv('../chipseq-wf/config/sampletable_SE.tsv', sep='\t', index=False)
#paired_end.to_csv('../chipseq-wf/config/sampletable_PE.tsv', sep='\t', index=False)

