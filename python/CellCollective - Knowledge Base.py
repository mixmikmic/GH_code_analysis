import cellcollective

sbml = cellcollective.load("https://cellcollective.org/#2329/apoptosis-network")

sbml.species

sbml.species_metadata("Cas3")

sbml.species_uniprotkb("Cas3")

sbml.species_ncbi_gene("Cas3")

lqm = cellcollective.to_biolqm(sbml)

