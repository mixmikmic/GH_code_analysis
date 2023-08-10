dbpath = "../testdata"

dbfile = "sample_basexdb"

import gcam_reader
import pandas as pd
import sys
import os.path

lcon = gcam_reader.LocalDBConn(dbpath, dbfile, suppress_gabble=False)

queries = gcam_reader.parse_batch_query(os.path.join('..','testdata','sample-queries.xml'))

[q.title for q in queries]

gdp_query = queries[3]
co2_query = queries[0]

print(gdp_query.title)
print(gdp_query.querystr)

lcon.runQuery(gdp_query)

rcom = gcam_reader.RemoteDBConn("sample_basexdb", "test", "test")

## To test running a query on a remote database, you have to start up
## a basex server on your local host and uncomment the next line.
##rcom.runQuery(co2_query)

allqueries = gcam_reader.importdata('../testdata/sample_basexdb', queries, scenarios='Reference-filtered')

allqueries.keys()

allqueries['Aggregated Land Allocation']



