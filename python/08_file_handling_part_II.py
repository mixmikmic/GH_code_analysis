directory = "res" 
import os 
csv_filename = os.path.join(directory, 'datafile.csv')
print csv_filename 

with open(csv_filename, 'r') as f: 
    for line in f: 
        print line.strip().split(',')    # let's save this in a list 

rows = [] 

with open(csv_filename) as f: 
    for line in f: 
        # print line.strip().split(',')   # let's save this in a list 
        rows.append(line.strip().split(','))
rows

for r in rows[1:]:       # skip the header 
    for c in range(2, 6): 
        r[c] = float(r[c])
rows

','.join( ['a', 'b', 'c'] )

rows 

for r in rows: 
    print [str(c) for c in r]   # save this in a list         

out_rows = []

for r in rows: 
    r = [str(c) for c in r]   # save this in a list 
    out_rows.append(r) 

out_rows

for r in out_rows: 
    out_line = ','.join(r)  # and then save this to a file 
    print out_line

csv_filename_out = os.path.join(directory, 'datafile-out.csv')
print csv_filename_out     

csv_filename_out = os.path.join(directory, 'datafile-out.csv')

csv_filename_out = os.path.join(directory, 'datafile-out.csv')
print csv_filename_out

with open(csv_filename_out, 'w') as f: 
    for r in out_rows: 
        out_line = ','.join(r)  # and then save this to a file 
        f.write(out_line + "\n")



