get_ipython().run_cell_magic('bq', 'query', "#standardSQL\n  --\n  -- Return variants for sample NA12878 that are:\n  --   annotated as 'pathogenic' or 'other' in ClinVar\n  --   with observed population frequency less than 5%\n  --\n  WITH sample_variants AS (\n  SELECT\n    -- Remove the 'chr' prefix from the reference name.\n    REGEXP_EXTRACT(reference_name, r'chr(.+)') AS chr,\n    start,\n    reference_bases,\n    alt,\n    call.call_set_name\n  FROM\n    `genomics-public-data.platinum_genomes_deepvariant.single_sample_genome_calls` v,\n    v.call call,\n    v.alternate_bases alt WITH OFFSET alt_offset\n  WHERE\n    call_set_name = 'NA12878_ERR194147'\n    -- Require that at least one genotype matches this alternate.\n    AND EXISTS (SELECT gt FROM UNNEST(call.genotype) gt WHERE gt = alt_offset+1)\n    ),\n  --\n  --\n  rare_pathenogenic_variants AS (\n  SELECT\n    -- ClinVar does not use the 'chr' prefix for reference names.\n    reference_name AS chr,\n    start,\n    reference_bases,\n    alt,\n    CLNHGVS,\n    CLNALLE,\n    CLNSRC,\n    CLNORIGIN,\n    CLNSRCID,\n    CLNSIG,\n    CLNDSDB,\n    CLNDSDBID,\n    CLNDBN,\n    CLNREVSTAT,\n    CLNACC\n  FROM\n    `bigquery-public-data.human_variant_annotation.ncbi_clinvar_hg38_20170705` v,\n    v.alternate_bases alt\n  WHERE\n    -- Variant Clinical Significance, 0 - Uncertain significance, 1 - not provided,\n    -- 2 - Benign, 3 - Likely benign, 4 - Likely pathogenic, 5 - Pathogenic,\n    -- 6 - drug response, 7 - histocompatibility, 255 - other\n    EXISTS (SELECT sig FROM UNNEST(CLNSIG) sig WHERE REGEXP_CONTAINS(sig, '(4|5|255)'))\n    -- TRUE if >5% minor allele frequency in 1+ populations\n    AND G5 IS NULL\n)\n --\n --\nSELECT\n  *\nFROM\n  sample_variants\nJOIN\n  rare_pathenogenic_variants USING(chr,\n    start,\n    reference_bases,\n    alt)\nORDER BY\n  chr,\n  start,\n  reference_bases,\n  alt")

