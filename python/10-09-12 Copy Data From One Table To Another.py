# Ignore
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite://')
get_ipython().run_line_magic('config', 'SqlMagic.feedback = False')

get_ipython().run_cell_magic('sql', '', "\n-- Create a table of criminals_1\nCREATE TABLE criminals_1 (pid, name, age, sex, city, minor);\nINSERT INTO criminals_1 VALUES (412, 'James Smith', 15, 'M', 'Santa Rosa', 1);\nINSERT INTO criminals_1 VALUES (234, 'Bill James', 22, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals_1 VALUES (632, 'Stacy Miller', 23, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals_1 VALUES (621, 'Betty Bob', NULL, 'F', 'Petaluma', 1);\nINSERT INTO criminals_1 VALUES (162, 'Jaden Ado', 49, 'M', NULL, 0);\nINSERT INTO criminals_1 VALUES (901, 'Gordon Ado', 32, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals_1 VALUES (512, 'Bill Byson', 21, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals_1 VALUES (411, 'Bob Iton', NULL, 'M', 'San Francisco', 0);")

get_ipython().run_line_magic('sql', 'SELECT * FROM criminals_1')

get_ipython().run_cell_magic('sql', '', '\nCREATE TABLE criminals_2 (pid, name, age, sex, city, minor);')

get_ipython().run_cell_magic('sql', '', '\nINSERT INTO criminals_2\nSELECT * FROM criminals_1')

get_ipython().run_cell_magic('sql', '', '\nSELECT * FROM criminals_2')

