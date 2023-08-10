sc

from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

#The clips are long, here is just a sample of the first entry
print dataset.data[0][:50] 

print len(dataset.data)

rdd = sc.parallelize(dataset.data)
print type(rdd)

terms = rdd.flatMap(lambda text: text.split())            .map(lambda word: (word.strip(".,-;?").lower(), 1))            .reduceByKey(lambda a, b: a+b)            .sortBy(lambda x: x[1], ascending=False)            .take(10)
            
print terms

