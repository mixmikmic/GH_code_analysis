import time
import sqlite3 as sql
import os

import sys
sys.path.append("/Users/kurner/Documents/classroom_labs")

class NoConnection(Exception):
    pass

class SQL_DB:  # a database
    
    # class level parameters
    backend  = 'sqlite3'
    user_initials  = 'KTU'
    timezone = int(time.strftime("%z", time.localtime()))
    target_path = "/Users/kurner/Documents/classroom_labs"  # current directory
    db_name = os.path.join(target_path, 'glossary.db')

    @staticmethod
    def mod_date():
        return time.mktime(time.gmtime())  # GMT time

    @classmethod
    def connect(cls):
        try:
            if cls.backend == 'sqlite3':
                cls.conn = sql.connect(cls.db_name)  # connection
                cls.curs = cls.conn.cursor()         # cursor
            elif cls.backend == 'mysql': # not using this, gives idea
                cls.conn = sql.connect(host='localhost',
                                      user='root', port='8889')
                cls.curs = cls.conn.cursor()
                
        except Exception:
            raise NoConnection
            
    @classmethod
    def disconnect(cls):
        cls.conn.close()

class DBcontext:
    """
    Generic parent class for connecting and disconnecting
    """

    def __init__(self, db):
        self.db = db      # references database class
        
    def __enter__(self):
        self.db.connect() 
        return self       # <-- for use inside with scope

    def __exit__(self, *stuff_happens):
        self.db.disconnect()
        if stuff_happens[0]:
            print("Exception raised!")
            print(stuff_happens)
            return True # <-- if considered handled, otherwise False
        return True

class Glossary(DBcontext):
    """
    Subclass with custom methods for this particular database
    """
    
    def create_table(self):
        # https://www.sqlite.org/lang_droptable.html
        self.db.curs.execute("""DROP TABLE IF EXISTS Glossary""")
        self.db.curs.execute("""CREATE TABLE Glossary
            (gl_term text PRIMARY KEY,
             gl_definition text,
             updated_at int,
             updated_by text)""")

    def save_term(self, *the_data):
        query = ("INSERT INTO Glossary "
        "(gl_term, gl_definition, updated_at, updated_by) "
        "VALUES ('{}', '{}', {}, '{}')".format(*the_data))
        # print(query)
        self.db.curs.execute(query)
        self.db.conn.commit()

with Glossary(SQL_DB) as dbx:  # <--- dbx returned by __enter__
    
    # for testing __exit__ in case of an exception
    # raise NoConnection
    
    dbx.create_table()
    FILE = os.path.join(dbx.db.target_path, "glossary.txt")
    
    with open(FILE, 'r', encoding='UTF-8') as gloss:
        lines = gloss.readlines()

    for line in lines:
        if len(line.strip()) == 0:
            continue
        term, definition = line.split(":", 1)
        right_now = dbx.db.mod_date()
        dbx.save_term(term[2:].strip(), definition.strip(), right_now, dbx.db.user_initials)

with Glossary(SQL_DB) as dbx:
    
    query = "SELECT gl_term, gl_definition FROM Glossary ORDER BY gl_term"
    
    dbx.db.curs.execute(query)  # gets the data
    
    print("{:^80}".format("GLOSSARY OF TERMS"))
    print("-" * 80)
    print("Term                                |Abbreviated Definition   " )
    print("-" * 80)
                           
    for term in dbx.db.curs.fetchmany(10):  # fetchone(), fetchmany(n), fetchall()
        print("{:35} | {:45}".format(term[0], term[1][:45]))

import chem_stuff

# modify database class to point to a different database file
SQL_DB.db_name = os.path.join(SQL_DB.target_path, 'periodic_table.db')

class ChemContext(DBcontext):
    """
    Subclass with custom methods for this particular database
    """
    
    def create_table(self):
        # https://www.sqlite.org/lang_droptable.html
        self.db.curs.execute("""DROP TABLE IF EXISTS Elements""")
        self.db.curs.execute("""CREATE TABLE Elements
            (elem_protons int PRIMARY KEY,
             elem_symbol text,
             elem_long_name text,
             elem_mass float,
             elem_series text,
             updated_at int,
             updated_by text)""")

    def save_term(self, *the_data):
        query = ("INSERT INTO Elements "
        "(elem_protons, elem_symbol, elem_long_name, elem_mass, elem_series,"
        "updated_at, updated_by) "
        "VALUES ({}, '{}', '{}', {}, '{}', {}, '{}')".format(*the_data))
        # print(query)
        self.db.curs.execute(query)
        self.db.conn.commit()
        
with ChemContext(SQL_DB) as dbx:
    
    dbx.create_table()
    
    FILE = os.path.join(dbx.db.target_path, "periodic_table.json")

    chem_stuff.load_elements(FILE)  # uses imported module to read JSON

    for atom in chem_stuff.all_elements.values():
        right_now = dbx.db.mod_date()
        dbx.save_term(atom.protons, atom.symbol, atom.long_name, atom.mass, atom.series,
                     right_now, dbx.db.user_initials)

with DBcontext(SQL_DB) as dbx:  # <--- dbx returned by __enter__
    
    query = ("SELECT elem_symbol, elem_long_name, elem_protons, elem_mass, elem_series" 
    " FROM Elements ORDER BY elem_protons")
    dbx.db.curs.execute(query)
    
    print("{:^70}".format("PERIODIC TABLE OF THE ELEMENTS"))
    print("-" * 70)
    print("Symbol |Long Name             |Protons |Mass   |Series  " )
    print("-" * 70)
   
    for the_atom in dbx.db.curs.fetchall():
        
        print("{:6} | {:20} | {:6} | {:5.2f} | {:15}".format(the_atom[0],
                          the_atom[1],
                          the_atom[2],
                          the_atom[3],
                          the_atom[4]))

