#start the SparkContext
from pyspark import SparkContext
sc = SparkContext(master="local[3]")

get_ipython().run_cell_magic('time', '', 'import urllib\ndata_dir=\'../../Data\'\nfilename=\'Moby-Dick.txt\'\nf = urllib.urlretrieve ("https://mas-dse-open.s3.amazonaws.com/"+filename, data_dir+\'/\'+filename)\n\n# First, check that the text file is where we expect it to be\n!ls -l $data_dir/$filename')

get_ipython().run_cell_magic('time', '', "text_file = sc.textFile(data_dir+'/'+filename)\ntype(text_file)")

get_ipython().run_cell_magic('time', '', 'counts = text_file.flatMap(lambda line: line.split(" ")) \\\n             .filter(lambda x: x!=\'\')\\\n             .map(lambda word: (word, 1)) \\\n             .reduceByKey(lambda a, b: a + b)\ntype(counts)')

print counts.toDebugString()

get_ipython().run_cell_magic('time', '', "Count=counts.count()\nSum=counts.map(lambda (w,i): i).reduce(lambda x,y:x+y)\nprint 'Count=%f, sum=%f, mean=%f'%(Count,Sum,float(Sum)/Count)")

get_ipython().run_cell_magic('time', '', 'C=counts.collect()\nprint type(C)')

C.sort(key=lambda x:x[1])
print 'most common words\n','\n'.join(['%s:\t%d'%c for c in C[-5:]])
print '\nLeast common words\n','\n'.join(['%s:\t%d'%c for c in C[:5]])

Count2=len(C)
Sum2=sum([i for w,i in C])
print 'count2=%f, sum2=%f, mean2=%f'%(Count2,Sum2,float(Sum2)/Count2)

get_ipython().run_cell_magic('time', '', "RDD=text_file.flatMap(lambda x: x.split(' '))\\\n    .filter(lambda x: x!='')\\\n    .map(lambda word: (word,1))")

get_ipython().run_cell_magic('time', '', 'RDD1=RDD.reduceByKey(lambda x,y:x+y)')

get_ipython().run_cell_magic('time', '', 'RDD2=RDD1.map(lambda (c,v):(v,c))\nRDD3=RDD2.sortByKey(False)')

print 'RDD3:'
print RDD3.toDebugString()

get_ipython().run_cell_magic('time', '', "C=RDD3.take(5)\nprint 'most common words\\n','\\n'.join(['%d:\\t%s'%c for c in C])")

