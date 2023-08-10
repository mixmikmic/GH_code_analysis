import tsne_fun

tsne_fun.normalize_and_make_tsne('phospho')

tsne_fun.normalize_and_make_tsne('phospho', qn_col=True)

tsne_fun.normalize_and_make_tsne('phospho', zscore_row=True)

tsne_fun.normalize_and_make_tsne('phospho', qn_col=True, zscore_row=True)

tsne_fun.normalize_and_make_tsne('phospho', filter_missing=True)

tsne_fun.normalize_and_make_tsne('phospho', qn_col=True, filter_missing=True)

tsne_fun.normalize_and_make_tsne('phospho', zscore_row=True, filter_missing=True)

tsne_fun.normalize_and_make_tsne('phospho', qn_col=True, zscore_row=True, filter_missing=True, skl_version=True)

