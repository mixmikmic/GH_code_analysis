# Ignore
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite://')
get_ipython().run_line_magic('config', 'SqlMagic.feedback = False')

get_ipython().run_cell_magic('sql', '', "\nCREATE TABLE criminals (pid INTEGER PRIMARY KEY AUTOINCREMENT,\n                        name, \n                        age, \n                        sex, \n                        city, \n                        minor);\n\nINSERT INTO criminals VALUES (NULL, 'James Smith', 15, 'M', 'Santa Rosa', 1);")

get_ipython().run_line_magic('sql', 'SELECT * FROM criminals')

get_ipython().run_cell_magic('sql', '', "\nINSERT INTO criminals VALUES (NULL, 'Bill James', 22, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (NULL, 'Stacy Miller', 23, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (NULL, 'Betty Bob', NULL, 'F', 'Petaluma', 1);\nINSERT INTO criminals VALUES (NULL, 'Jaden Ado', 49, 'M', NULL, 0);\nINSERT INTO criminals VALUES (NULL, 'Gordon Ado', 32, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (NULL, 'Bill Byson', 21, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (NULL, 'Bob Iton', NULL, 'M', 'San Francisco', 0);")

get_ipython().run_line_magic('sql', 'SELECT * FROM criminals')

