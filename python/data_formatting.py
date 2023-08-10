from os import listdir
from os.path import isfile, join

col_names = ['Label']
for x in map(chr,range(65,81)):
    for y in map(str,range(1,9)):
        col_names.append('Sensor_'+x+y)
col_names.append('Batch_No\n')
print 'Number of columns -',len(col_names)

out=open('formatted_data.csv','w')
out.write(','.join(col_names))

raw_files = [f for f in listdir('./raw_data') if isfile(join('./raw_data', f))]

for file_name in raw_files:
    with open('./raw_data/'+file_name,'r') as f:
        for i in f:
            j=i.strip().split(' ')
            out.write(','.join([j[0]]+[k.split(':')[1] for k in j[1:]]+[file_name.strip('batch').split('.')[0],'\n']))
out.close()

#for file_name in raw_files:
#    with open('./raw_data/'+file_name,'r') as f:
#        for i in f:
#            j=i.strip().split(' ')
#            target_label = [j[0]]
#            attributes = [k.split(':')[1] for k in j[1:]]
#            batch_no = [file_name.strip('batch').split('.')[0],'\n']
#            out.write(','.join(target_label+attributes+batch_no))
#out.close()

