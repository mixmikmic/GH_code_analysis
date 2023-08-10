from io import BytesIO
from zipfile import ZipFile
from urllib import request
import datetime as dt

start = dt.datetime.now()

url = request.urlopen('http://www-odi.nhtsa.dot.gov/downloads/folders/Complaints/FLAT_CMPL.zip')
zipfile_in_memory = ZipFile(BytesIO(url.read()))
zipfile_in_memory.extractall(r'D:\temp')
zipfile_in_memory.close()
print("Download and extraction completed")

columns = [
    'CMPLID',
    'ODINO',
    'MFR_NAME',
    'MAKETXT',
    'MODELTXT',
    'YEARTXT',
    'CRASH',
    'FAILDATE',
    'FIRE',
    'INJURED',
    'DEATHS',
    'COMPDESC',
    'CITY',
    'STATE',
    'VIN',
    'DATEA',
    'LDATE',
    'MILES',
    'OCCURENCES',
    'CDESCR',
    'CMPL_TYPE',
    'POLICE_RPT_YN',
    'PURCH_DT',
    'ORIG_OWNER_YN',
    'ANTI_BRAKES_YN',
    'CRUISE_CONT_YN',
    'NUM_CYLS',
    'DRIVE_TRAIN',
    'FUEL_SYS',
    'FUEL_TYPE',
    'TRANS_TYPE',
    'VEH_SPEED',
    'DOT',
    'TIRE_SIZE',
    'LOC_OF_TIRE',
    'TIRE_FAIL_TYPE',
    'ORIG_EQUIP_YN',
    'MANUF_DT',
    'SEAT_TYPE',
    'RESTRAINT_TYPE',
    'DEALER_NAME',
    'DEALER_TEL',
    'DEALER_CITY',
    'DEALER_STATE',
    'DEALER_ZIP',
    'PROD_TYPE',
    'REPAIRED_YN',
    'MEDICAL_ATTN',
    'VEHICLES_TOWED_YN'
]

import sqlite3
import pandas as pd
import datetime as dt

conn = sqlite3.connect(r'D:\NHTSA\nhtsa.db')
cursor = conn.cursor()

# Since we are going to load/re-create the complaint's table in its entirety, DROP it
cursor.execute('DROP TABLE IF EXISTS complaints')

chunksize = 20000
j = 0

begin = dt.datetime.now()

# use the columns list to define the column names of the complaints table
for df in pd.read_csv(r'D:\temp\FLAT_CMPL.txt', names=columns, dtype=object, chunksize=chunksize, 
                      delimiter='\t', iterator=True, encoding='ISO-8859-1', error_bad_lines=False):    
    j+=1
    # To print on same line, use '\r' and end='' option with the print function
    print('\r'+'{} seconds: completed {} rows'.format((dt.datetime.now() - begin).seconds, j*chunksize),end='')

    df.to_sql('complaints', conn, if_exists='append', index=False)
cursor.close()
conn.close()

conn = sqlite3.connect(r'D:\NHTSA\nhtsa.db')
cursor = conn.cursor()

cursor.execute('CREATE INDEX make ON complaints (MAKETXT)')
cursor.execute('CREATE INDEX addeddate ON complaints (DATEA)')
cursor.execute('CREATE INDEX faildate ON complaints (FAILDATE)')
cursor.execute('CREATE INDEX compdesc ON complaints (COMPDESC)')
cursor.execute('CREATE INDEX "make-faildate" ON complaints (MAKETXT, FAILDATE)')
cursor.execute('CREATE INDEX "year-make-model" ON complaints (MAKETXT, MODELTXT, YEARTXT)')

cursor.close()
conn.close()

print("Total elapsed time (hr:min:sec):", dt.datetime.now() - start)

