import sqlite3

conn = sqlite3.connect('/mnt/c/Users/Aumit/Desktop/mxm_dataset_train.txt/mxm_dataset.db')

res = conn.execute("SELECT * FROM sqlite_master WHERE TYPE='table'")

res.fetchall()

res = conn.execute("SELECT word FROM words")

len(res.fetchall())

res = conn.execute("SELECT word FROM words WHERE ROWID=4703")

res.fetchone()[0]

conn_tmdb = sqlite3.connect("/mnt/c/Users/Aumit/Desktop/mxm_dataset_train.txt/track_metadata.db")

res = conn.execute("SELECT track_id FROM lyrics WHERE word='pretty'")

len(res.fetchall())

res = conn.execute("SELECT track_id FROM lyrics WHERE word='pretti'")

len(res.fetchall())

