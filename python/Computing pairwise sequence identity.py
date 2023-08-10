from skbio import DNA
from skbio.alignment import global_pairwise_align_nucleotide
s1 = DNA("GAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGCATGCTTAACACATGCAAGTCGAACGGCAGCATGACTTAGCTTGCTAAGTTGATGGCGAGTGGCGAACGGGTGAGTAACGCGTAGGAATATGCCTTAAAGAGGGGGACAACTTGGGGAAACTCAAGCTAATACCGCATAAACTCTTCGGAGAAAAGCTGGGGACTTTCGAGCCTGGCGCTTTAAGATTAGCCTGCGTCCGATTAGCTAGTTGGTAGGGTAAAGGCCTACCAAGGCGACGATCAGTAGCTGGTCTGAGAGGATGACCAGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGGACAATGGGGGCAACCCTGATCCAGCAATGCCGCGTGTGTGAAGAAGGCCTGAGGGTTGTAAAGCACTTTCAGTGGGGAGGAGGGTTTCCCGGTTAAGAGCTAGGGGCATTGGACGTTACCCACAGAAGAAGCACCGGCTAACTCCGTGCCAGCAGCCCGCGGTAATACGGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCCGTTAAAAGGTGCCTAAGGTGGTTTGGATAGTTATGTGTTAAATTCCCTGGCGCCTCCACCCTGGGCCAGGTCCATATAAAAACTGTTAAACTCCGAAGTATGGGCACAAGGTAATTGGAAATTCCGGTGGTACCGTGAAAATGCGCTTAGAGATCGGGAAGGGACCACCCCAGTGGGGAAGGCGGCTACCTGGCCTAATAACTGACATTGAGGCACGAAAAGCGTGGGGAGCAACCAGGATTAGATACCCTGGTAGTCCACGCTGTAAACGATGTCAACTAGCTGTGGTTATATGAATATAATTAGTGGCGAAGCTAACGCGATAAGTTGACCGCCTGGGGAGTACGGTCGCAAGATTAAAACTCAAAGGAATGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGATGCAACGCGAAGAACCTTACCTACCCTTGACATACAGTAAATCTTTCAGAGATGAGAGAGTGCCTTCGGGAATACTGATACAGGTGCTGCATGGCTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGTAACGAGCGCAACCCTTATCTCTAGTTGCCAGCGAGTAATGTCGGGAACTCTAAAGAGACTGCCGGTGACAAACCGGAGGAAGGCGGGGACGACGTCAAGTCATCATGGCCCTTACGGGTAGGGCTACACACGTGCTACAATGGCCGATACAGAGGGGCGCGAAGGAGCGATCTGGAGCAAATCTTATAAAGTCGGTCGTAGTCCGGATTGGAGTCTGCAACTCGACTCCATGAAGTCGGAATCGCTAGTAATCGCGAATCAGCATGTCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCATGGGAGTGGGCTGCACCAGAAGTAGATAGTCTAACCGCAAGGGGGACGTTTACCACGGTGTGGTTCATGACTGGGGTGAAGTCGTAACAAGGTAGCCG")
s2 = DNA("TTTTCTTGGATTTGATTCTGGTCCAGAGTAAACGCTTGAGATATGTTGATACATGTTAGTTAAACGTGAATATTTGGTTTTTATGCCAACTTTATTTAAGTAGCGTATAGGTGAGTAATATGCAAGAATCCTACCTTTTAGTTTATGTAGCTCGTAAATTTATAAAAGATTTTTTCGCTAAAAGATGGGCTTGCACAAGATTAGGTTTTTGGTTTGCTAAAAACGTTCCAAGCCTAAGATCTTTAGCCGGCTTTCGTGAGTGACCGGCCACATAGGGACTGAGACAATGCCCTAGCTCCTTTTCTGGAGGCATCAGTACAAAGCATTGGACAATGAACGAAAGTTTGATCCAGTAATATCTCGTGAATGATGAAGGGTTTTTGCTCGTAAATTTCTTTTAGTTGAAAGAAAAAAGATATATTTCAACAGAAAAAATCCTGGCAAATCCTCGTGCCAGCAGCCGCGGTAATACGAGAAGGGTTAGCGTTACTCGAAATTATTGGGCGTAAAGTGCGTGAACAGCTGCTTTTTAAGCTATAGGCAGAAAAATCAAGGGTTAATCTTGTTTTTGTCATAGTTCTGATAAGCTTGAGTTTGGAAGAAGATAATAGAACATTTTATGGAGCGATGAAATGCTATGATATAAAAGAGAATACCAAAAGCGAAGGCAGTTATCTAGTACAAAACTGACGCCTATACGCGAAGGCTTAGGTAGCAAAAAGGATTAGGGACCCTTGTAGTCTAAGCTGTCAACGATGAACACTCGTTTTTGGATCACTTTTTTTCAGAAACTAAGCTAACGCGTTAAGTGTTTCGCCTGGGTACTACGGTCGCAAGACTAAAACTTAAAGAAATTGGCGGGAGTAAAAACAAGCAGTGGAGCGTGTGGTTTAATTCGATAGTACACGCAAATCTTACCATTACTTGACTCAAACATTGAAATGCACTATGTTTATGGTGTTGTTTAAGTATTATTTTACTTATAGATGTGCAGGCGCTGCATGGTTGTCGTCAGTTCGTGTCGTGAGATGTTTGGTTAATTCCCTTAACGAACGTAACCCTCAAAGCATATTCAAAACATTTTGTTTTTTTGTTAAACAGTCGGGGAAACCTGAATGTAGAGGGGTAGACGTCTAAATCTTTATGGCCCTTATGTATTTGGGCTACTCATGCGCTACAATGGGTGTATTCTACAAAAAGACGCAAAAACTCTTCAGTTTGAGCAAAACTTGAAAAGCACCCTCTAGTTCGGATTGAACTCTGGAACTCGAGTTCATAAAGTTGGAATTGCTAGTAATCGTGAGTTAGCGTATCGCGGTGAATCGAAAATTTACTTTGTACATACCGCCCGTCAAGTACTGAAAATTTGTATTGCAAGAAATTTTTGGAGAATTTACTTAACTCTTTTTTTTTTTAAGTTGGCTGTATCAGTCTTTTAAAAACTTTGAGTTAGGTTTTAAGCATCCGAGGGTAAAAGCAACATTTTTTATTGGTATTAAGTCGTAACAAGGTAGCCCTACGGG")

aln, _, _ = global_pairwise_align_nucleotide(s1, s2, penalize_terminal_gaps=True)

seq1, seq2 = aln
seq1.match_frequency(seq2, relative=True)

seq1.match_frequency(seq2)

