import sys
from random import random
from operator import add

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonPi").getOrCreate()

#partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
partitions = 10
n = 100000 * partitions

def f(_):
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 <= 1 else 0

count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
print("Pi is roughly %f" % (4.0 * count / n))

spark.stop()

sourceBytes = '                                                                                         \nimport sys                                                                                              \nfrom random import random                                                                               \nfrom operator import add                                                                                \n                                                                                                        \nfrom pyspark.sql import SparkSession                                                                    \n                                                                                                        \nspark = SparkSession.builder.appName("PythonPi").getOrCreate()                                          \n                                                                                                        \npartitions = 10                                                                                         \nn = 100000 * partitions                                                                                 \n                                                                                                        \ndef f(_):                                                                                               \n    x = random() * 2 - 1                                                                                \n    y = random() * 2 - 1                                                                                \n    return 1 if x ** 2 + y ** 2 <= 1 else 0                                                             \n                                                                                                        \ncount = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)                  \nprint("Pi is roughly %f" % (4.0 * count / n))                                                           \n                                                                                                        \nspark.stop()                                                                                            \n'.encode('utf-8')

print(sourceBytes.decode("utf-8"))

# This can be removed once the Docker file for jupyterhb is intact
get_ipython().system('cd ~ && rm scripts && ln -s /root/pipeline/jupyterhub.ml/scripts')

get_ipython().system('ls ~/scripts/pi')

get_ipython().system('mkdir -p ~/scripts/pi/')

with open('/root/scripts/pi/pi.py', 'wb') as f:
  f.write(sourceBytes)

get_ipython().system('cat ~/scripts/pi/pi.py')

get_ipython().system('git status')

get_ipython().system('git add --all ~/scripts')

get_ipython().system('git status')

get_ipython().system('git commit -m "updated pyspark scripts"')

get_ipython().system('git status')

# If this fails with "Permission denied", use terminal within jupyter to manually `git push`
get_ipython().system('git push')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from IPython.display import clear_output, Image, display, HTML

html = '<iframe width=100% height=500px src="http://demo.pipeline.io:8080/admin">'
display(HTML(html))



