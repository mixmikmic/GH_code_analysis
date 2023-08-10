get_ipython().magic('run notebook.config.ipy')

statement = '''select plate, row, column, fraction_spike, cufflinks_no_genes_pc as no_genes from qc_summary'''
df = DB.fetch_DataFrame(statement,db)

get_ipython().run_cell_magic('R', '-i df -w 700 -h 400', '\n# Order the columns\ndf$column <- factor(df$column, levels=c(1:12))\n\n# Make plots\ngp <- ggplot(df, aes(column, row)) + geom_tile(aes(fill = no_genes))\ngp <- gp + scale_fill_gradient(low="white", high = "red", na.value="grey", name="Number of genes detected") \ngp <- gp + geom_point(aes(size=fraction_spike))\ngp <- gp + facet_wrap(~plate)\ngp <- gp + labs(size = \'Fraction spike-ins\')\ngp <- gp + ggtitle("Number of genes detected & fraction spike-ins")\ngp <- gp + xlab("")+ ylab("")\nprint(gp)\n\n#ggsave(plot = gp, filename="number_genes_fraction_spike_per_plate.pdf", device=cairo_pdf)')

