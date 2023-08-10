import sqlite3

connection = sqlite3.connect("example.db")

cursor = connection.cursor()

try:
    cursor.execute("DROP TABLE Point") # Let's clean up the database just in case
except Exception:
    pass

cursor.execute("CREATE TABLE Point (x NUMBER, y NUMBER)")
cursor.execute("INSERT INTO Point VALUES (1, 2)")
cursor.execute("INSERT INTO Point VALUES (2, 4)")
cursor.execute("INSERT INTO Point VALUES (-2, 3)")

cursor.execute("SELECT * FROM Point")
results = cursor.fetchall()
print results

cursor.close()
connection.close()

