array = range(1,1000)

p  =sc.parallelize(array)

q = p.map(lambda x: x*x)

q.take(5)

demo = sc.textFile('../data/Demographic_Statistics_By_Zip_Code.csv').cache()
demo.take(2)

cols = demo.take(1)
print cols[0].split(',')

headers = [item.replace(' ', '_') for item in cols[0].split(',')]

raw_rows = demo.filter(lambda line: not line.startswith('JURISDICTION NAME,')).map(lambda row: row.split(','))
raw_rows.take(3)

head_b = sc.broadcast(headers)
from pyspark.sql import Row

#this is a very naive implementation and should not be used in prod
def detect_data(data):
    try:
        return int(data)
    except ValueError:
        pass
    try:
        return float(data)
    except ValueError:
        return data

def raw_to_row(row):
    clean_row = [detect_data(value) for value in row]
    values_dict = dict(zip(head_b.value, clean_row))
    return Row(**values_dict)

schema_rows = raw_rows.map(raw_to_row).toDF()
schema_rows.take(3)

sqlCtx.registerDataFrameAsTable(schema_rows, 'demographics')
sqlCtx.cacheTable('demographics')

sqlCtx.sql('select MAX(PERCENT_FEMALE) from demographics').collect()

sqlCtx.sql('select JURISDICTION_NAME from demographics where PERCENT_FEMALE > 0.90').collect()

sqlCtx.tableNames()

demo = sqlCtx.table('demographics')

demo.write.format('parquet').saveAsTable('demographics_parquet')



