fout = open("data.txt","w")

for i in range(0,5):
    fout.write( "Current value of i is %d\n" % i )  # note, we add a newline with \n at the end of each line

fout.close()

fout.write("More data!")

fin = open("data.txt","r")  # action "r" means open file to read

lines = fin.readlines()
for l in lines:
    print( l.strip() )  # note that we usually need to remove the newline characters from the end of strings

fin.close()

fout = open("simple.csv","w")
# create the records
for row in range(5):
    # start the record with an identifier
    fout.write("record_%d" % (row+1) )
    # create the fields for each record
    for col in range(4):
        value = (row+1)*(col+1)     # just create some dummy values
        fout.write(",%d" % value )  # notice the comma separator
    # move on to a new line in the file
    fout.write("\n")
# finished, so close the file
fout.close()    

fin = open("simple.csv","r")
print( fin.read() )
fin.close()

fin = open("simple.csv","r")
# process the file line by line
for line in fin.readlines():
    # remove the newline character from the end
    line = line.strip()
    # split the line based on the comma separator
    parts = line.split(",")
    # extract the identifier as the first value in the list
    record_id = parts[0]
    # convert the rest to integers from strings
    values = []
    for s in parts[1:]:
        values.append( int(s) )
    # display the record
    print( record_id, values )
# finished, so close the file
fin.close()

