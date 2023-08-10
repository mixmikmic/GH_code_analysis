from pprint import pprint
# I, Python am built from types, such as builtin types:

the_builtins = dir(__builtins__) # always here

pprint(the_builtins[-10:])  # no need to import

for the_string in ["list", "tuple", "dict", "int", "float"]:
    if the_string in the_builtins:
        print("Yes I am a native type: ", the_string)
        assert type(eval(the_string)) == type # all types in this club
    else:
        print("No, I'm not native: ", the_string)

# usually up top
from string import ascii_lowercase as all_lowers
from random import shuffle

class P:
    """
    class Px is the more sophisticated version of this class
    """
    def __init__(self, p=None):
        if not p:
            original = all_lowers + ' '
            scrambled = list(original)
            shuffle(scrambled)            
            self.perm = dict(zip(original, scrambled))
        else:
            self.perm = p
        
    def __invert__(self):
        """reverse my perm, make a new me"""
        reverse = dict(zip(self.perm.values(), self.perm.keys()))
        return P(reverse)  # <-- new P instance
        
    def encrypt(self, s):
        output = ""
        for c in s:
            output += self.perm[c]
        return output
            
    def decrypt(self, s):
        rev = ~self  # <-- new P instance
        return rev.encrypt(s) # <-- symmetric key

     
p = P()
m = "i like python so much because it does everything" # palindrome
c = p.encrypt(m)
print(m)  # plaintext
print(c)  # ciphertext
d = p.decrypt(c)
print(d)

import sqlite3 as sql
import os.path
import json
import time
from contextlib import contextmanager

PATH = "/Users/kurner/Documents/classroom_labs/session10"
DB1 = os.path.join(PATH, 'periodic_table.db')

def mod_date():
    return time.mktime(time.gmtime())  # GMT time

@contextmanager        
def Connector(db):
    try:
        db.conn = sql.connect(db.db_name)  # connection
        db.curs = db.conn.cursor()   # cursor
        yield db       
    except Exception as oops:
        if oops[0]:
            raise
    db.conn.close()

class elemsDB:
    
    def __init__(self, db_name):
        self.db_name = db_name
     
    def seek(self, elem):
        if self.conn:
            if elem != "all":
                query = ("SELECT * FROM Elements "
                "WHERE elem_symbol = '{}'".format(elem))
                self.curs.execute(query)
                result = self.curs.fetchone()
                if result:
                    return json.dumps(list(result))
            else:
                query = "SELECT * FROM Elements ORDER BY elem_protons"
                self.curs.execute(query)
                result={}
                for row in self.curs.fetchall():
                    result[row[1]] = list(row)
                return json.dumps(result)                
        return "NOT FOUND"

output = ""
with Connector(elemsDB(DB1)) as dbx:
    output = dbx.seek("S")

print(output)

import requests

data = {}
data["protons"]=92
data["symbol"]="U"
data["long_name"]="Uranium"
data["mass"]=238.02891
data["series"]="Dunno"
data["secret"]="DADA" # <--- primitive authentication

# the_url = 'http://localhost:5000/api/elements'
the_url = 'http://thekirbster.pythonanywhere.com/api/elements'
r = requests.post(the_url, data=data)
print(r.status_code)
print(r.content)



