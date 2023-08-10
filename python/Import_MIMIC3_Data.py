import csv
import os
import sys
import sqlite3


# the files to read
mimic_files = ('DIAGNOSES_ICD_DATA_TABLE.csv',
               'D_ICD_PROCEDURES_DATA_TABLE.csv',
               'PATIENTS_DATA_TABLE.csv',
               'D_ICD_DIAGNOSES_DATA_TABLE.csv', 
               'NOTEEVENTS_DATA_TABLE.csv',
               'DRGCODES_DATA_TABLE.csv',
               'D_CPT_DATA_TABLE.csv',
               'CPTEVENTS_DATA_TABLE.csv')

sqlitedb = os.path.join(os.path.expanduser('~'),'Box Sync', 'GradSchoolStuff', 'MastersProject', 'mimic3', 'mimic3.sqlite')
if (os.path.exists(sqlitedb)):
    print("Database already exists - proceed with caution!")
    sys.exit()

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists DIAGNOSES_ICD;
    CREATE TABLE DIAGNOSES_ICD
    (   ROW_ID INT NOT NULL,
        SUBJECT_ID INT NOT NULL,
        HADM_ID INT NOT NULL,
        SEQ_NUM INT,
        ICD9_CODE VARCHAR(20),
        CONSTRAINT diagnosesicd_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists D_ICD_PROCEDURES;
    CREATE TABLE D_ICD_PROCEDURES
    (   ROW_ID INT NOT NULL,
        ICD9_CODE VARCHAR(10) NOT NULL,
        SHORT_TITLE VARCHAR(50) NOT NULL,
        LONG_TITLE VARCHAR(255) NOT NULL,
        CONSTRAINT d_icd_proc_code_unique UNIQUE (ICD9_CODE),
        CONSTRAINT d_icd_proc_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists PATIENTS;
    CREATE TABLE PATIENTS
    (   ROW_ID INT NOT NULL,
        SUBJECT_ID INT NOT NULL,
        GENDER VARCHAR(5) NOT NULL,
        DOB TIMESTAMP(0) NOT NULL,
        DOD TIMESTAMP(0),
        DOD_HOSP TIMESTAMP(0),
        DOD_SSN TIMESTAMP(0),
        EXPIRE_FLAG INT NOT NULL,
        CONSTRAINT pat_subid_unique UNIQUE (SUBJECT_ID),
        CONSTRAINT pat_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists D_ICD_DIAGNOSES;
    CREATE TABLE D_ICD_DIAGNOSES
    (   ROW_ID INT NOT NULL,
        ICD9_CODE VARCHAR(10) NOT NULL,
        SHORT_TITLE VARCHAR(50) NOT NULL,
        LONG_TITLE VARCHAR(255) NOT NULL,
        CONSTRAINT d_icd_diag_code_unique UNIQUE (ICD9_CODE),
        CONSTRAINT d_icd_diag_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists NOTEEVENTS;
    CREATE TABLE NOTEEVENTS
    (   ROW_ID INT NOT NULL,
        SUBJECT_ID INT NOT NULL,
        HADM_ID INT,
        CHARTDATE TIMESTAMP(0),
        CHARTTIME TIMESTAMP(0),
        STORETIME TIMESTAMP(0),
        CATEGORY VARCHAR(50),
        DESCRIPTION VARCHAR(255),
        CGID INT,
        ISERROR CHAR(1),
        TEXT TEXT,
        CONSTRAINT noteevents_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists DRGCODES;
    CREATE TABLE DRGCODES
    (   ROW_ID INT NOT NULL,
        SUBJECT_ID INT NOT NULL,
        HADM_ID INT NOT NULL,
        DRG_TYPE VARCHAR(20) NOT NULL,
        DRG_CODE VARCHAR(20) NOT NULL,
        DESCRIPTION VARCHAR(255),
        DRG_SEVERITY SMALLINT,
        DRG_MORTALITY SMALLINT,
        CONSTRAINT drg_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists D_CPT;
    CREATE TABLE D_CPT
    (   ROW_ID INT NOT NULL,
        CATEGORY SMALLINT NOT NULL,
        SECTIONRANGE VARCHAR(100) NOT NULL,
        SECTIONHEADER VARCHAR(50) NOT NULL,
        SUBSECTIONRANGE VARCHAR(100) NOT NULL,
        SUBSECTIONHEADER VARCHAR(255) NOT NULL,
        CODESUFFIX VARCHAR(5),
        MINCODEINSUBSECTION INT NOT NULL,
        MAXCODEINSUBSECTION INT NOT NULL,
        CONSTRAINT dcpt_ssrange_unique UNIQUE (SUBSECTIONRANGE),
        CONSTRAINT dcpt_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop table if exists CPTEVENTS;
    CREATE TABLE CPTEVENTS
    (   ROW_ID INT NOT NULL,
        SUBJECT_ID INT NOT NULL,
        HADM_ID INT NOT NULL,
        COSTCENTER VARCHAR(10) NOT NULL,
        CHARTDATE TIMESTAMP(0),
        CPT_CD VARCHAR(10) NOT NULL,
        CPT_NUMBER INT,
        CPT_SUFFIX VARCHAR(5),
        TICKET_ID_SEQ INT,
        SECTIONHEADER VARCHAR(50),
        SUBSECTIONHEADER VARCHAR(255),
        DESCRIPTION VARCHAR(200),
        CONSTRAINT cpt_rowid_pk PRIMARY KEY (ROW_ID)
    );''')

connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.execute('select * from sqlite_master')
    row = cursor.fetchone()
    while row:
        print(row)
        row = cursor.fetchone()

for mf in mimic_files:
    file = os.path.join(os.path.expanduser('~'), 'Box Sync', 'GradSchoolStuff', 'MastersProject', 'mimic3', mf)

    if not (os.path.exists(file)):
        print("Specified file does not exist")
        sys.exit()

    csvReader = csv.reader(open(file, newline=''))
    header = next(csvReader)
    print('Columns read from ', mf, ':', header)

    table_name = mf.replace('_DATA_TABLE.csv', '')
    print('Loading to ', table_name)
    
    value_placeholder = ('?,'*len(header))[:-1]
    
    ## load each line from CSV into appropriate table
    connection = sqlite3.connect(sqlitedb)
    with connection:
        for row in csvReader:
            cursor = connection.cursor()
            cursor.execute('insert into ' + table_name + ' values (' + value_placeholder + ')', row)

# now that all the tables are created and data loaded - create indexes
# Got these indexes from 
#    https://github.com/MIT-LCP/mimic-code/blob/master/buildmimic/postgres/postgres_add_indexes.sql
connection = sqlite3.connect(sqlitedb)
with connection:
    cursor = connection.cursor()
    cursor.executescript('''
    drop index IF EXISTS NOTEEVENTS_idx01;
    CREATE INDEX NOTEEVENTS_idx01
        ON NOTEEVENTS (SUBJECT_ID);
    drop index IF EXISTS NOTEEVENTS_idx02;
    CREATE INDEX NOTEEVENTS_idx02
        ON NOTEEVENTS (HADM_ID);
    drop index IF EXISTS NOTEEVENTS_idx05;
    CREATE INDEX NOTEEVENTS_idx05
        ON NOTEEVENTS (CATEGORY);
    
    drop index IF EXISTS DIAGNOSES_ICD_idx01;
    CREATE INDEX DIAGNOSES_ICD_idx01
        ON DIAGNOSES_ICD (SUBJECT_ID);
    drop index IF EXISTS DIAGNOSES_ICD_idx02;
    CREATE INDEX DIAGNOSES_ICD_idx02
        ON DIAGNOSES_ICD (ICD9_CODE);
    drop index IF EXISTS DIAGNOSES_ICD_idx03;
    CREATE INDEX DIAGNOSES_ICD_idx03
        ON DIAGNOSES_ICD (HADM_ID);
        
    drop index IF EXISTS CPTEVENTS_idx01;
    CREATE INDEX CPTEVENTS_idx01
        ON CPTEVENTS (SUBJECT_ID);
    drop index IF EXISTS CPTEVENTS_idx02;
    CREATE INDEX CPTEVENTS_idx02
        ON CPTEVENTS (CPT_CD);
        
    drop index IF EXISTS DRGCODES_idx01;
    CREATE INDEX DRGCODES_idx01
        ON DRGCODES (SUBJECT_ID);
    drop index IF EXISTS DRGCODES_idx02;
    CREATE INDEX DRGCODES_idx02
        ON DRGCODES (DRG_CODE);
    drop index IF EXISTS DRGCODES_idx03;
    CREATE INDEX DRGCODES_idx03
        ON DRGCODES (DESCRIPTION);
    ''')



