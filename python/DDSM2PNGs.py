__version__   = '0.0.1'
__status__    = 'Development'

import os, sys
import re
import pandas as pd

def read_ics(dir_name, fname):
    fpath = os.path.join(dir_name, fname)
    
    try: 
        with open(fpath) as fobj:
            for line in fobj:
                if line.startswith('filename'):
                    fn = (line.split()[1]).replace('-', '_')
                if line.startswith('DIGITIZER'):
                    digitizer = line.split()[1]
                if line.startswith('LEFT_CC'):
                    LEFT_CC_rows = line.split()[2]
                    LEFT_CC_cols = line.split()[4]
                if line.startswith('LEFT_MLO'):
                    LEFT_MLO_rows = line.split()[2]
                    LEFT_MLO_cols = line.split()[4]
                if line.startswith('RIGHT_CC'):
                    RIGHT_CC_rows = line.split()[2]
                    RIGHT_CC_cols= line.split()[4]
                if line.startswith('RIGHT_MLO'):
                    RIGHT_MLO_rows = line.split()[2]
                    RIGHT_MLO_cols= line.split()[4]
            
            meta = [
                {'fn': fn+'.LEFT_CC.LJPEG',   'rows': LEFT_CC_rows,   'cols': LEFT_CC_cols,   'digitizer': digitizer},
                {'fn': fn+'.LEFT_MLO.LJPEG',  'rows': LEFT_MLO_rows,  'cols': LEFT_MLO_cols,  'digitizer': digitizer},
                {'fn': fn+'.RIGHT_CC.LJPEG',  'rows': RIGHT_CC_rows,  'cols': RIGHT_CC_cols,  'digitizer': digitizer},
                {'fn': fn+'.RIGHT_MLO.LJPEG', 'rows': RIGHT_MLO_rows, 'cols': RIGHT_MLO_cols, 'digitizer': digitizer},
            ]
        
            return pd.DataFrame(meta)
    except IOError as error:
        print('File Error Encountered: {0} {1}'.format(fname, error[1]))
    except UnboundLocalError as error:
        print ('In directory {0}, problem encountered with DDSM files. Check directory.'.format(dir_name))
        sys.exit(1)

def create_PNGs(start_dir, DDSM_dir):
    
    os.chdir(DDSM_dir)
    for dir_name, subdir_list, file_list in os.walk(DDSM_dir):
        for fname in file_list:
            if re.match(r'.*[.]ics', fname):
                print 'In directory: {0}\t ... found the file {1}'.format(dir_name, fname)
                meta = read_ics(dir_name, fname)
                print meta[['fn', 'rows', 'cols', 'digitizer']], '\n'
                # Call functions to generate 4 PNM files per case using Dr. Chris Rose's utilities jpeg.exe & ddsmraw2pnm.exe
                # For each PNM file, generate a PNG file using NETpbm's pnmtopng function
                
    os.chdir(start_dir)
    

start_dir = os.getcwd()

create_PNGs(start_dir, '/Users/jnarhan/Code/CUNY7_CapStone/DDSM_Images/cases' )




