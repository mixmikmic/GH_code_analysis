get_ipython().system('pip install Biopython')

input_path = 'Test_sequences/test_file.fasta'
# Try test_file_1.fasta or test_file_2.fasta in the next line
# If you want to use your own files, put the full URL of publicy accessible file here.
# Don't forget to change organism_subgroup accordingly, and don't leave it empty or too
# vague (like Bacteria) - BLAST will timeout if the amount of work is too high.
input_url = 'https://github.com/NCBI-Hackathons/NCBI_Jupyter/raw/master/NoteBooks/TestData/test_file.fasta'
organism_subgroup = '"Staphylococcus aureus"[orgn]'
save_blast_run = False
# If you set save_blast_run to True, make sure that you have writeable directory
# Test_sequences in your Jupyther space, otherwise intermediate results can't be saved
blast_result_path = 'Test_sequences/blast.xml'
use_best_homogeneity_filter = True

from Bio.Blast.NCBIWWW import qblast

#f = open(input_path)
import urllib
f = urllib.request.urlopen(input_url)
data = f.read()

res = qblast('blastn', 'nr', data, entrez_query=organism_subgroup)

if save_blast_run:
    res_str = res.read()
    with open(blast_result_path,'w') as f:
        f.write(res_str)

from Bio.Blast import NCBIXML

def best_organisms(result_handle, use_best_homogeneity_filter):
    seqs_to_total_best_bits = {}
    seqs_to_def = {}
    blast_records = NCBIXML.parse(result_handle)
    for blast_record in blast_records:
        best_bits = 0
        best_id = ""
        best_def = ""
        best_hits = []
        for aln in blast_record.alignments:
            # This is an alignment between blast_record.query_id and aln.hit_id
            total_bits = 0
            total_query_len = 0
            total_identity = 0
            for hsp in aln.hsps:
                if hsp.expect < E_VALUE_THRESH:
                    total_bits += hsp.bits
                    total_query_len += hsp.query_end - hsp.query_start + 1
                    total_identity  += hsp.identities
            if total_query_len == 0: continue
            best_hits.append((total_bits, total_identity/total_query_len, aln.hit_id, aln.hit_def))
            if total_bits > best_bits:
                best_bits = total_bits
                best_id   = aln.hit_id
                best_def  = aln.hit_def
        best_hits.sort(reverse=True)
        # Here we have all hits of a given contig, blast_record.query_id sorted in
        # best-first order
        # We can analyze if the several best hits are close to each other and thus
        # don't differentiate our organism well. We sort out such contigs.
        if use_best_homogeneity_filter:
            if len(best_hits) == 0: continue
            if len(best_hits) > 1:
                # if the first 2 hits are too close to each other and their identity score the same
                # throw the contig away
                if (best_hits[0][0]-best_hits[1][0])/best_hits[0][0] < 0.01 and                    (best_hits[0][1]-best_hits[1][1])/best_hits[0][1] < 0.01:
                    continue
                best_bits = best_hits[0][0]
                best_id   = best_hits[0][2]
                best_def  = best_hits[0][3]
        if best_id == "": continue
#        seqs_to_best_hits.setdefault(best_id, []).append(best_bits)
        seqs_to_def[best_id] = best_def
        seqs_to_total_best_bits[best_id] = seqs_to_total_best_bits.get(best_id, 0) + best_bits
    return seqs_to_total_best_bits, seqs_to_def

E_VALUE_THRESH = 0.04
# This dict maps sequence id to best hits to it
#seqs_to_best_hits = {}
seqs_to_total_best_bits = {}
seqs_to_def = {}
if save_blast_run:
    with open(blast_result_path) as result_handle:
        seqs_to_total_best_bits, seqs_to_def = best_organisms(result_handle, use_best_homogeneity_filter)
else:
    seqs_to_total_best_bits, seqs_to_def = best_organisms(res, use_best_homogeneity_filter)

sorted_organisms = []
for k, bits in seqs_to_total_best_bits.items():
    sorted_organisms.append((bits, k))
sorted_organisms.sort(reverse=True)
print('Best organisms are:')
seq_id = sorted_organisms[0][1] # Best id
print(seqs_to_def[seq_id], 'with total score:', seqs_to_total_best_bits[seq_id])
seq_id = sorted_organisms[1][1] # Second best id
print(seqs_to_def[seq_id], 'with total score:', seqs_to_total_best_bits[seq_id])



