import os
import glob
from jinja2 import Template
import pandas as pd
import random

# read in the base_rx keyfile template using jinja2 templating
with open(os.path.join('Rx_Template','Base_Rx.key'), 'r') as base_keyfile:
    template = Template(base_keyfile.read())
    print('Found Base_Rx.key and created jinja2 template.')

inserts = {}

inserts['FVSIn'] = 'FVSIn'
inserts['FVSOut'] = 'FVSOut'

template_kcps = glob.glob(os.path.join('Rx_Template','*.kcp'))
if len(template_kcps) > 0:
    print('Found the following kcp files in the Rx_Template directory:')
    for kcp in template_kcps:
        fname = os.path.split(kcp)[-1]
        print(fname, end='...')
        # read the kcp file
        key = fname.split('.')[0] # key for item in inserts dictionary
        with open(kcp, 'r') as item:
            value = item.read()
        # add the contents of the kcp file to the inserts dictionary
        inserts[key] = value
        print(' added to template.')
else: 
    raise FileNotFoundError('No kcp files found in the Rx_Template directory.')

SDImax_dict = {} # variant: SDImax keywords
logvalue_dict = {} # variant: log value specs

variant_kcps = glob.glob(os.path.join('Rx_Template','variant_specific_kcps','*.kcp'))
if len(variant_kcps) > 0:
    print('Found the following kcp files in the variant_specific_kcps directory:')
    for kcp in variant_kcps:
        fname = os.path.split(kcp)[-1]
        variant = fname.split('.')[0][-2:]
        kcp_type = fname.split('.')[0][:-3]
        print(variant, kcp_type, end='...')
        # read the kcp file
        with open(kcp, 'r') as item:
            key = variant
            value = item.read()
        if kcp_type == 'SDImax':
            SDImax_dict[key] = value
        elif kcp_type == 'logvalue':
            logvalue_dict[key] = value
        print(' added to template.')
else: 
    raise FileNotFoundError('No kcp files found in the variant_specific_kcps directory.')

rxs_dict = {}
# a dictionary storing the silvicultural keywords for each rx
rx_kcps = glob.glob(os.path.join('Rx_Template', 'Rxs', 'rx*.kcp'))
if len(rx_kcps) > 0:
    print('Found the following kcp files in the Rxs subdirectory:')
    for kcp in rx_kcps:
        fname = os.path.split(kcp)[-1]
        print(fname, end='...')
        # read the kcp file
        key = fname.split('/')[-1].split('.')[0] # key for item in inserts dictionary
        with open(kcp, 'r') as item:
            value = item.read()
        # add the contents of the kcp file to the inserts dictionary
        rxs_dict[key] = value
        print(' added to template.')
else: 
    raise FileNotFoundError('No kcp files found in the Rx_Template directory.')

def create_keyfile(standID, variant, rx, offset):
    '''
    Creates a single FVS keyfile based on the jinja2 template.
    '''
    inserts['ID'] = standID
    inserts['SDImax'] = SDImax_dict[variant]
    inserts['Log_Value'] = logvalue_dict[variant]
    inserts['rx'] = rxs_dict[rx]
    inserts['offset'] = offset
    # FVS slows down outputting to large databases, so we'll divide output among 10 databases
    inserts['db_num'] = random.randint(1,10) # add a random number, 1-10 for a output database suffix
    
    fname = 'fvs'+variant+'_stand'+str(standID)+'_'+rx+'_off'+str(offset)+'.key'
    path = os.path.abspath('keyfiles_to_run')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join('keyfiles_to_run',fname),'w') as keyfile:
        keyfile.write(template.render(**inserts))

def create_keyfiles(stands, variants, rxs, offsets=[0], verbose=False):
    '''
    Creates FVS keyfiles for all stands using Base_Rx.key as a template.
    Arguments:
    stands: List of standIDs that keyfiles will be created for. Required.
    variants: List of 2-letter codes of FVS variant for each stand. Required.
    rxs: a list of rx names to build keyfiles for. Required.
    offsets: optional, a list of offsets, used in FVS to delay implementation of a 
        management regime. e.g., [0, 5, 10]. Defaults to a list with no offsets (i.e., [0]).
    '''
    stands_processed = 0
    keyfiles_written = 0
    num_stands = len(stands)
    num_keys = num_stands * len(rxs) * len(offsets)
    print('Creating {:,} keyfiles for {:,} stands.'.format(num_keys, num_stands))

    if not verbose:
        print('Stands processed', end=": ")
    for i in range(len(stands)):
        if verbose:
            print('Creating keyfiles for stand', stands[i], end='... ')
        stand_keyfiles = 0
        for rx in rxs:
            for offset in offsets:
                # run the create_keyfile function
                create_keyfile(standID=stands[i], variant=variants[i], rx=rx, offset=offset) 
                keyfiles_written += 1
                stand_keyfiles += 1
        stands_processed += 1
        if verbose:
            print(stand_keyfiles, 'keyfiles written.')
        else:
            if stands_processed % 100 == 0:
                print('{:,}'.format(stands_processed), end='... ')
    print('Done. Created', keyfiles_written, 'keyfiles for', stands_processed, 'stands.')

# option to import from CSV file with standID and variant
# stands = pd.read_csv(os.path.join('Data','Inner_Riparian.csv'))

import psycopg2

conn = psycopg2.connect("dbname='FVSIn' user='postgres' host='localhost'") # password in pgpass file
SQL = '''
SELECT stand_id, variant
FROM stand_init;
'''
# read the query into a pandas dataframe
stands = pd.read_sql(SQL, conn)

# close the database connection
conn.close()

get_ipython().run_cell_magic('time', '', "rxs_to_run = ['rx1', 'rx2', 'rx3', 'rx4', 'rx5']\ncreate_keyfiles(stands=stands.stand_id, variants=stands.variant, rxs=rxs_to_run, verbose=False)")

