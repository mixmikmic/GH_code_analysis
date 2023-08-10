import pydna
from Bio.Restriction import BamHI, EcoRV
from gel import Gel, Sample, Q_, randDNAseqs, ladders, ladder_from_info, lindivQ

# For convenience
def printQ(quantities):
    for Q in quantities:
        print Q

gb = pydna.Genbank("pg25220@alunos.uminho.pt") # Tell Genbank who you are!

gene = gb.nucleotide("X06997") # Kluyveromyces lactis LAC12 gene for lactose permease.

primer_f, primer_r = pydna.parse(''' >760_KlLAC12_rv (20-mer)
                                     ttaaacagattctgcctctg

                                     >759_KlLAC12_fw (19-mer)
                                     aaatggcagatcattcgag
                                     ''', ds=False)

pcr_prod = pydna.pcr(primer_f, primer_r, gene)

vector = gb.nucleotide("AJ001614") # pCAPs cloning vector

lin_vector = vector.linearize(EcoRV)

rec_vec = (lin_vector + pcr_prod).looped()

print len(pcr_prod)
print len(rec_vec)
print len(lin_vector)

sample1 = Sample([pcr_prod, rec_vec, lin_vector])
print repr(sample1)
print sample1

sample2 = Sample(randDNAseqs([500, 1000, 5000]))
sample3 = Sample(randDNAseqs([3000, 1500]))
print repr(sample2)
print repr(sample3)

samples = [sample1, sample2, sample3]
G = Gel(samples)
gelpic = G.run()
gelpic.show()
printQ(G.quantities)

ladders.keys()

ladder = ladder_from_info('1kb_GeneRuler')
print ladder

samples = [sample1, sample2, sample3, ladder]
G = Gel(samples)
gelpic = G.run()
gelpic.show()
printQ(G.quantities)

dseqs3 = randDNAseqs([3000, 1500])
qts3 = [100, 100]
sample3 = Sample(dseqs3, qts3, Q_(14, 'ul'))
print sample3

dseqs2 = randDNAseqs([500, 1000, 5000])
qts2 = lindivQ(dseqs2, 200)
sample2 = Sample(dseqs2, qts2, Q_(14, 'ul'))
print sample2

samples = [ladder, sample1, sample2, sample3]
G = Gel(samples)
gelpic = G.run()
gelpic.show()
printQ(G.quantities)

