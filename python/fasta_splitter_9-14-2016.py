from Bio import SeqIO
handle = open("/home/bay001/projects/kes_20160307/permanent_data/kestrel5-reclustered.no-aerv.no-mtdna.fasta", "rU")
outpref = "/home/bay001/projects/kes_20160307/permanent_data/parts/part_"
part = 0
count = 0
records = []
for record in SeqIO.parse(handle, "fasta"):
    part = part + 1
    records.append(record)
    if(part%1000 == 0):
        count = count + 1
        SeqIO.write(records,outpref+"{}.fas".format(count),"fasta")
        records = []
handle.close()





