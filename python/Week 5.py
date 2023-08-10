import sqlite3
sqlite_file = '/Users/MelodyHuang/Desktop/db1.sqlite'
"/Users/MelodyHuang/Desktop/db2.sqlite"
#If such a file does not exist prior to this, this will automatically create the file
conn = sqlite3.connect(sqlite_file) 
c = conn.cursor()

table_name1 = 'table1'
table_name2 = 'table2'
field1 = 'column1'
field_type1 = 'INTEGER'

#Creates a new table with 1 column
c.execute('CREATE TABLE {tn} ({nf} {ft})'.format(tn = table_name1, nf=field1, ft=field_type1))

#Adds column without a row value: 
field2 = 'column2' 
field_type2 = 'TEXT'
field3 = 'column3'
field_type3 = "TEXT"
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"          .format(tn=table_name1, cn=field2, ct = field_type2))

#If we want to set a default row value: 
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct} DEFAULT '{df}'"         .format(tn = table_name1, cn = field3, ct = field_type3, df = "Hello World"))

#Create new table: 
c.execute('CREATE TABLE {tn} ({nf} {ft} PRIMARY KEY)'.          format(tn = table_name2, nf=field1, ft=field_type1))
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"          .format(tn=table_name2, cn=field2, ct = field_type2))

#Insert values into a second column 
try:
    c.execute("INSERT INTO {tn} ({idf}, {cn}) VALUES (123456, 'test')".             format(tn = table_name2, idf = field1, cn = field2))
except sqlite3.IntegrityError: 
    print("ERROR: ID already exists in PRIMARY KEY column{}".format(id_column))

#We can do the above in a more compact manner: 
c.execute("INSERT OR IGNORE INTO {tn} ({idf}, {cn}) VALUES (123456, 'test')".        format(tn=table_name2, idf=field1, cn=field2))

#Updating an entry: 
c.execute("UPDATE {tn} SET {cn} = ('Hello World!') WHERE {idf} = (123456)".         format(tn = table_name2, idf = field1, cn = field2))

#Equivalent to df$field2[which(df$field1 == 123456)] = "Updated Value"

#Selecting ALL columns for row that match a value: 
c.execute('SELECT * FROM {tn} WHERE {cn}="Hello World!"'.          format(tn=table_name2, cn=field2))
all_rows = c.fetchall()
print('1):', all_rows)
#THIS IS EQUIVALENT TO THE R SYNTAX: df[which(df$field2 == "Hi World",]

#Selecting a select column for rows that match a value: 
c.execute('SELECT ({coi}) FROM {tn} WHERE {cn}="Hello World!"'.        format(coi=field1, tn=table_name2, cn=field2))
all_rows = c.fetchall()
print('2):', all_rows)

#Commits changes we've made to the database
conn.commit()
conn.close() #Closes the connection

