import sqlite3

#connect to database
con = sqlite3.connect('PQRS.sqlite3')

# generate a cursor
cur = con.cursor()

# define a SQL query string
create_table_query = """ CREATE TABLE product (
                        name text NOT NULL, 
                        price text NOT NULL
                    )"""

# execute the query
cur.execute(create_table_query)

# Creating a database
import sqlite3  
con = sqlite3.connect('PQRS.sqlite3')  
cur = con.cursor()
create_table_query = """ CREATE TABLE product (
                    name text NOT NULL, 
                    price text NOT NULL
                )"""

# take user input
name = input("Enter name")
price = input("Enter price")

# generate query based on user input
add_data_query = """INSERT INTO product (name, price) VALUES ('{}', '{}')""".format(name, price)

# execute the query
cur.execute(create_table_query)
cur.execute(add_data_query)

# commit or save the execution
con.commit()

#close the cursor
cur.close()

import sqlite3  
con = sqlite3.connect('PQRS.sqlite3')  
cur = con.cursor()
add_data_query = """INSERT INTO product (name, price) VALUES ('Test Product 123', '50000')"""
cur.execute(add_data_query)
con.commit()
cur.close()

import sqlite3  
con = sqlite3.connect('PQRS.sqlite3')  
cur = con.cursor()
con.commit()
cur.execute("SELECT * FROM product")
print(cur.fetchone())  #fetch just one item (this behaves as a generator)

print(cur.fetchall()) # fetch all records in the database


print(cur.fetchmany(1)) #Fetches several rows from the resultset.

import sqlite3

class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

conn = sqlite3.connect('product.db')

c = conn.cursor()

c.execute("""CREATE TABLE products (
            name text,
            price float
            )""")


def insert_product(prod):
    with conn: # work with context manager so that we don't have to commit everytime
        # paramaterized SQL query
        c.execute("INSERT INTO products VALUES (:name, :price)", {'name': prod.name, 'price': prod.price})


def get_product_by_name(name):
    c.execute("SELECT * FROM products WHERE name=:name", {'name': name})
    return c.fetchall()


def update_price(prod, price):
    with conn:
        c.execute("""UPDATE products SET price = :price
                    WHERE name = :name""",
                  {'name': prod.name, 'price': price})


def remove_product(prod):
    with conn:
        c.execute("DELETE from products WHERE name = :name",
                  {'name': prod.name})

p1 = Product("DSLR Camera", 50000)
p2 = Product("Toshiba Laptop", 750000)
p3 = Product("ADATA External HDD 1TB", 7500)

insert_product(p1)
insert_product(p2)
insert_product(p3)

prod = get_product_by_name('DSLR Camera')
print(prod)

update_price(p2, 70000)

remove_product(p3)

conn.close()

