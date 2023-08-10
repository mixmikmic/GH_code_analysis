from jp_gene_viz import dExpression

dExpression.load_javascript_support()

E = dExpression.display_heat_map("expr.tsv", show=True)

E.match_text.value = "th0*"
E.match_click()
E.genes_text.value = "nsf nme4 nudt4"
E.genes_click()
E.title_html.value = "experiment th0 values for ssf, nme4, and nudt4"
E.transform_dropdown.value = dExpression.LOG2_TRANSFORM



