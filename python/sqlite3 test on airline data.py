ls

import sqlite3
conn = sqlite3.connect("flights.db")

cur = conn.cursor()

cur.execute("select * from airlines limit 5;")

results = cur.fetchall()
print(results)

cur.close()   # Close the connection
conn.close()

import pandas as pd
import sqlite3

conn = sqlite3.connect("flights.db")
df = pd.read_sql_query("select * from airlines limit 5;", conn)
df

df["country"]

cur = conn.cursor()
cur.execute("insert into airlines values (6048, 19846, 'Test flight', '', '', null, null, null, 'Y')")

conn.commit()

pd.read_sql_query("select * from airlines where id=19846;", conn)

cur = conn.cursor()
values = ('Test Flight', 'Y')
cur.execute("insert into airlines values (6049, 19847, ?, '', '', null, null, null, ?)", values)
conn.commit()

cur = conn.cursor()
values = ('USA', 19847)
cur.execute("update airlines set country=? where id=?", values)
conn.commit()

pd.read_sql_query("select * from airlines where id=19847;", conn)

from datetime import datetime
df = pd.DataFrame(
    [[1, datetime(2016, 9, 29, 0, 0) , datetime(2016, 9, 29, 12, 0), 'T1', 1]], 
    columns=["id", "departure", "arrival", "number", "route_id"]
)
df

df.to_sql("daily_flights", conn, if_exists="replace")

pd.read_sql_query("select * from daily_flights;", conn)

df = pd.read_sql("select * from daily_flights", conn)
df["delay_minutes"] = None
df.to_sql("daily_flights", conn, if_exists="replace")



