from skbio import DNA

dna = DNA.read('data/single_sequence1.fasta', seq_num=1)
dna

purine_runs = list(dna.find_motifs('purine-run', min_length=2))
longest_purine = max(purine_runs, key=lambda x: x.stop - x.start)

dna[longest_purine]

longest_purine

pyrimidine_runs = list(dna.find_motifs('pyrimidine-run', min_length=2))
longest_pyrimidine = max(pyrimidine_runs, key=lambda x: x.stop - x.start)

dna[longest_pyrimidine]

longest_pyrimidine

t_runs = list(dna.find_with_regex('([T]+)'))
longest_t = max(t_runs, key=lambda x: x.stop - x.start)

dna[longest_t]

longest_t

