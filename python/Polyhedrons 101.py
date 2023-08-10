import sqlite3 as sql
import os
from pprint import pprint

class DB:
    backend  = 'sqlite3'       # default
    target_path = os.getcwd()  # current directory
    db_name = ":file:"         # lets work directly with a file
    db_name = os.path.join(target_path, 'shapes_lib.db')

    @classmethod
    def connect(cls):
        if cls.backend == 'sqlite3':
            DB.conn = sql.connect(DB.db_name)
            DB.c = DB.conn.cursor()
        elif cls.backend == 'postgres': # or something else
            DB.conn = sql.connect(host='localhost',
                                  user='root', port='8889')
            DB.c = DB.conn.cursor()

    @classmethod
    def disconnect(cls):
        DB.conn.close()

DB.connect()
DB.c.execute("SELECT poly_long, poly_color, poly_volume from Polys") # query
pprint(DB.c.fetchall()) # print results
DB.disconnect()

DB.connect()
DB.c.execute("SELECT vertex_label, coord_a, coord_b, coord_c, coord_d FROM Coords ORDER BY vertex_label") # query
pprint(DB.c.fetchall()) # print results
DB.disconnect()

DB.connect()
DB.c.execute("SELECT vertex_label, coord_x, coord_y, coord_z FROM Coords ORDER BY vertex_label") # query
pprint(DB.c.fetchall()) # print results
DB.disconnect()

