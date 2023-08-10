import sqlite3 # psycopg2 # pymysql 

conn = sqlite3.connect('example.sqlite3')

cur = conn.cursor()

cur.execute('CREATE TABLE countries(id integer, name text, iso3 text)')

cur.execute('SELECT * FROM countries')

cur.fetchall()

cur.execute('INSERT INTO countries(id, name, iso3) VALUES(1, "Nepal", "NEP")')

cur.execute('SELECT * FROM countries')

cur.fetchall()

sql = '''INSERT INTO countries(id, name, iso3) VALUES(?, ?, ?)'''
cur.executemany(sql, [(2, 'India', 'INA'), 
                      (3, 'Bhutan', 'BHU'), 
                      (4, 'Afganistan', 'AFG')])

cur.execute('SELECT * FROM countries')

cur.fetchall()

sql = 'INSERT INTO countries(id, name, iso3) VALUES(4, "Pakistan", "PAK")'
cur.execute(sql)

cur.execute('SELECT * FROM countries')
cur.fetchall()

sql = 'UPDATE countries SET id=5 WHERE iso3="PAK"'
cur.execute(sql)

cur.execute('SELECT * FROM countries')
cur.fetchall()

sql = 'UPDATE countries SET id=5'
cur.execute(sql)

cur.execute('SELECT * FROM countries')
cur.fetchall()

conn.commit()

cur.execute('SELECT * FROM countries')
cur.fetchall()

cur.execute('SELECT * FROM countries WHERE id=4')
cur.fetchall()

cur.execute('SELECT * FROM countries WHERE id>3')
cur.fetchall()

cur.execute('SELECT * FROM countries WHERE name LIKE "%an"')
cur.fetchall()

cur.execute('SELECT * FROM countries WHERE name LIKE "%an%"')
cur.fetchall()

cur.execute('SELECT * FROM countries WHERE name LIKE "an%"')
cur.fetchall()

cur.execute('DELETE FROM countries')

cur.execute('SELECT * FROM countries')
cur.fetchall()

conn.commit()

import csv

sql = 'INSERT INTO countries(id, name, iso3) VALUES(?, ?, ?)'
_id = 1
with open('untitled.txt', 'r') as datafile:
    csvfile = csv.DictReader(datafile)
    for row in csvfile:
        if row['euname'] and row['iso3']:
            cur.execute(sql, (_id, row['euname'], row['iso3']))
            _id += 1
conn.commit()

cur.execute('SELECT * FROM countries')
cur.fetchall()

sql = '''CREATE TABLE 
country_list(id integer primary key autoincrement,
country_name text not null,
iso3 text not null unique)'''
cur.execute(sql)

sql = 'INSERT INTO country_list(country_name, iso3) VALUES(?, ?)'
with open('untitled.txt', 'r') as datafile:
    csvfile = csv.DictReader(datafile)
    for row in csvfile:
        if row['euname'] and row['iso3']:
            cur.execute(sql, (row['euname'], row['iso3']))
conn.commit()

cur.execute('SELECT * FROM country_list')
cur.fetchall()

cur.execute('''INSERT INTO country_list(id, country_name, iso3)
VALUES(47, 'Cuba', 'CCB')''')

cur.execute('SELECT * FROM country_list')
cur.fetchall()

class Book:
    id = None
    name = None
    isbn = None
    
    def __init__(self, name, isbn):
        self.name = name
        self.isbn = isbn
    
    def save(self):
        if self.id:
            cur.execute('UPDATE books SET name=?,isbn=? WHERE id=?',
                        (self.name, self.isbn, self.id))
        else:
            cur.execute('INSERT INTO books(name, isbn) VALUES(?, ?)',
                        (self.name, self.isbn))
    
    @staticmethod
    def get_books_by_name(name):
        cur.execute('SELECT * FROM books WHERE name LIKE "%?%', 
                    (name,))
        return cur.fetchall()
    
    @staticmethod
    def get_all_books():
        pass
    
    @staticmethod
    def get_book_by_id(_id):
        cur.execute('SELECT * FROM books WHERE id=?', (_id,))
        result = cur.fetchone()
        if result:
            book = Book()
            book.id, book.name, book.isbn = result
            return book
        return None

book1 = Book('Learn nepali', 'akshdkajjsdhk')
book1.save()

Book.get_books_by_name('Learning')

book2 = Book.get_book_by_id(4)
book2.name = 'New Name'
book2.save()











