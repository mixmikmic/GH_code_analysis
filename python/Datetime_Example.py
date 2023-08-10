#Unfortunatel, this won't work on Windows.
get_ipython().system('head sample_data.csv')

data_tuples = list()
with open('sample_data.csv','r') as f:
    for line in f:
        data_tuples.append(line.strip().split(','))

data_tuples[0:10]

#Figure out the format string
# http://pubs.opengroup.org/onlinepubs/009695399/functions/strptime.html 
import datetime
x = '2016-01-01 00:00:09'
format_str = "%Y-%m-%d %H:%M:%S"
datetime.datetime.strptime(x,format_str)

data_tuples = list()
with open('sample_data.csv','r') as f:
    for line in f:
        data_tuples.append(line.strip().split(','))
        
        
import datetime
for i in range(0,len(data_tuples)):
    data_tuples[i][0] = datetime.datetime.strptime(data_tuples[i][0],format_str)
    data_tuples[i][1] = float(data_tuples[i][1])

#Let's see if this worked
data_tuples[0:10]

#Extract the hour from a datetime object
x=data_tuples[0][0]
x.hour

data_tuples = [(x[0].hour,x[1]) for x in data_tuples]

data_tuples = list()
with open('sample_data.csv','r') as f:
    for line in f:
        data_tuples.append(line.strip().split(','))
import datetime
for i in range(0,len(data_tuples)):
    data_tuples[i][0] = datetime.datetime.strptime(data_tuples[i][0],format_str)
    data_tuples[i][1] = float(data_tuples[i][1])

def get_data(filename):
    data_tuples = list()
    with open(filename,'r') as f:
        for line in f:
            data_tuples.append(line.strip().split(','))
    import datetime
    format_str = "%Y-%m-%d %H:%M:%S"
    data_tuples = [(datetime.datetime.strptime(x[0],format_str).hour,float(x[1])) for x in data_tuples]
    return data_tuples    

get_data('sample_data.csv')

buckets = dict()
for item in get_data('sample_data.csv'):
    if item[0] in buckets:
        buckets[item[0]][0] += 1
        buckets[item[0]][1] += item[1]
    else:
        buckets[item[0]] = [1,item[1]]

buckets

for key,value in buckets.items():
    print("Hour:",key,"\tAverage:",value[1]/value[0])

def get_hour_bucket_averages(filename):
    def get_data(filename):
        data_tuples = list()
        with open(filename,'r') as f:
            for line in f:
                data_tuples.append(line.strip().split(','))
        import datetime
        format_str = "%Y-%m-%d %H:%M:%S"
        data_tuples = [(datetime.datetime.strptime(x[0],format_str).hour,float(x[1])) for x in data_tuples]
        return data_tuples        
    buckets = dict()
    for item in get_data(filename):
        if item[0] in buckets:
            buckets[item[0]][0] += 1
            buckets[item[0]][1] += item[1]
        else:
            buckets[item[0]] = [1,item[1]]  
    return [(key,value[1]/value[0]) for key,value in buckets.items()]

get_hour_bucket_averages('sample_data.csv')

