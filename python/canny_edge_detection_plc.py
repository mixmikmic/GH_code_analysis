get_ipython().magic("run 'database_connectivity_setup.ipynb'")

get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns
import StringIO
from array import array
from matplotlib import pyplot as plt
from PIL import Image

sql = """
        SELECT 
            column_name, 
            data_type 
        FROM 
            information_schema.columns 
        WHERE 
            table_schema = 'iot' and table_name = 'src_image'
"""
psql.read_sql(sql,conn)

sql = """SELECT count(*) FROM iot.src_image;"""
psql.read_sql(sql,conn)

sql = """DROP FUNCTION IF EXISTS iot.CannyEdgeDetectPLC(varchar) CASCADE;"""
psql.execute(sql,conn)

sql = """
        CREATE or REPLACE FUNCTION iot.CannyEdgeDetectPLC(varchar) 
        RETURNS int[]
        AS 
            '/usr/local/lib/ds/canny_plc.so', 'canny_plc'
        language C strict immutable;
"""
psql.execute(sql,conn)

sql = """DROP FUNCTION IF EXISTS iot.GetImgSizePLC(varchar) CASCADE;"""
psql.execute(sql,conn)

sql = """
        CREATE or REPLACE FUNCTION iot.GetImgSizePLC(varchar)
        RETURNS int[]
        AS
            '/usr/local/lib/ds/canny_plc.so', 'get_imgsize'
        language C strict immutable;
"""
psql.execute(sql,conn)
conn.commit()

sql = """DROP TABLE IF EXISTS iot.edges_table;"""
psql.execute(sql,conn)

sql = """
        CREATE TABLE iot.edges_table AS
            SELECT
                img_name,
                iot.GetImgSizePLC(img) as imgsize, 
                iot.CannyEdgeDetectPLC(img) as edges
            FROM iot.src_image;
"""
psql.execute(sql,conn)
conn.commit();

sql = """SELECT count(*) FROM iot.edges_table;"""
psql.read_sql(sql,conn)

sql = """
        SELECT
            img_name,
            imgsize,
            edges
        FROM
            iot.edges_table
        WHERE img_name like '%Images/CBIR/Caltech256/256_ObjectCategories/009.bear/009_0049.jpg%'
        LIMIT 1;        
"""
df = psql.read_sql(sql,conn)

imgname = np.array(df.loc[0]['img_name'])
size = np.array(df.loc[0]['imgsize'])
edges = np.array(df.loc[0]['edges'])
edges = np.reshape(edges, (size[0],size[1]))
plt.imshow(edges)
plt.axis('off')
plt.show()

sql = """
        SELECT
            img_name,
            img
        FROM
            iot.src_image
        WHERE img_name like '%Images/CBIR/Caltech256/256_ObjectCategories/009.bear/009_0049.jpg%'
        LIMIT 1;        
"""
df = psql.read_sql(sql,conn)

imgname = np.array(df.loc[0]['img_name'])

img = df.loc[0]['img']
buf = img.split(',')
buf = map(int, buf)
buf = array('b',buf)
buf = buf.tostring()

buff = StringIO.StringIO() 
buff.write(buf)
buff.seek(0)
im = Image.open(buff)
plt.imshow(im)
plt.axis('off')
plt.show()

conn.close()

