# Ignore
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite://')
get_ipython().run_line_magic('config', 'SqlMagic.feedback = False')

get_ipython().run_cell_magic('sql', '', "\n-- Create a table of criminals\nCREATE TABLE criminals (pid, name, age, sex, city, minor);\nINSERT INTO criminals VALUES (412, 'James Smith', 15, 'M', 'Santa Rosa', 1);\nINSERT INTO criminals VALUES (234, 'Bill James', 22, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (632, 'Stacy Miller', 23, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (621, 'Betty Bob', NULL, 'F', 'Petaluma', 1);\nINSERT INTO criminals VALUES (162, 'Jaden Ado', 49, 'M', NULL, 0);\nINSERT INTO criminals VALUES (901, 'Gordon Ado', 32, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (512, 'Bill Byson', 21, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (411, 'Bob Iton', NULL, 'M', 'San Francisco', 0);")

get_ipython().run_cell_magic('sql', '', "\n-- Select everything\nSELECT *\n\n-- From the table 'criminals'\nFROM criminals")

