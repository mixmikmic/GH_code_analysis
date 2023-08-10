get_ipython().system(' find /repositories')

from sourced.engine import Engine
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder  .master("local[*]").appName("Examples")  .getOrCreate()

engine = Engine(spark, "/repositories/siva/latest/*/", "siva")

engine.repositories.select('id').distinct().show(10, False)

engine.repositories.printSchema()

engine.repositories.references.head_ref.show()

engine.repositories.references.head_ref.select('repository_id', 'hash').show(10, False)

repos = engine.repositories
head_refs = repos.references.head_ref
tree_entries = head_refs.commits.tree_entries
readmes = tree_entries.filter(tree_entries.path == 'README.md')
contents = readmes.blobs.collect()

for row in contents:
    print(row.repository_id)
    lines = [l.decode("utf-8") for l in row.content.splitlines()]
    for (i, line) in enumerate(lines):
        if len(line) == 0:
            continue
        if line[0] == '#':
            print(line)
            break
        if line[0] == '=':
            print(lines[i-1])
            break
    print('')

