## Imports

import sqlite3

DBPATH = 'people.db'
conn = sqlite3.connect(DBPATH)

cursor = conn.cursor()

sql = (
    "CREATE TABLE IF NOT EXISTS companies ("
    "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "    name TEXT NOT NULL"
    ")"
)

cursor.execute(sql)

# Create the contacts table

sql = "INSERT INTO companies (name) VALUES (?)"
cursor.execute(sql, ("Georgetown University",))
conn.commit()

cursor.execute(sql, ("US Department of Commerce",))
conn.commit()

cursor.execute("SELECT id FROM companies WHERE name=?", ("Georgetown University",))
print cursor.fetchone()

# Insert some contacts and companies using the methods described above.

import os
import sqlite3

def create_tables(conn):
    """
    Write your CREATE TABLE statements in this function and execute
    them with the passed in connection. 
    """
    # TODO: fill in. 
    pass


def connect(path="people.db", syncdb=False):
    """
    Connects to the database and ensures there are tables.
    """
    
    # Check if the SQLite file exists, if not create it.
    if not os.path.exists(path):
        syncdb=True

    # Connect to the sqlite database
    conn = sqlite3.connect(path)
    if syncdb:
        create_tables(conn)
    
    return conn


def insert(name, email, company, conn=None):
    if not conn: conn = connect()

    # Attempt to select company by name first. 
    
    # If not exists, insert and select new id.
    
    # Insert contact

    
if __name__ == "__main__":
    name    = raw_input("Enter name: ")
    email   = raw_input("Enter email: ")
    company = raw_input("Enter company: ")
    
    conn = connect()
    insert(name, email, company, conn)

    # Change below to count contacts per company! 
    contacts = conn.execute("SELECT count(id) FROM contacts").fetchone()
    print "There are now {} contacts".format(*contacts)

    conn.close()

