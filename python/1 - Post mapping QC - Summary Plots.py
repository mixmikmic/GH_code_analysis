get_ipython().magic('run notebook.config.ipy')

statement = '''select * from qc_summary'''

df = DB.fetch_DataFrame(statement,db)

id_columns = ["sample_id"] + PARAMS["name_field_titles"].split(",")

data = pd.melt(df,id_vars=id_columns)
#data.head()

get_ipython().run_cell_magic('R', '-i data -w 1000 -h 1200', '\ngp <- ggplot(data, aes(x=value))\ngp <- gp + facet_wrap(~variable, scales="free")\ngp <- gp + geom_histogram(nbin=200)\ngp <- gp + theme(axis.text.x=element_text(angle=90))\ngp <- gp + ggtitle("Post-mapping QC histograms\\n")\ngp <- gp + ylab("no. cells") + xlab("value - fraction or count")\n\nsuppressMessages(print(gp))\n#ggsave("post_mapping_qc.pdf", gp, device=cairo_pdf)')

get_ipython().run_cell_magic('R', '-i id_columns -i df -w1000 -h600', '\nmat <- as.matrix(df[,!colnames(df) %in% id_columns])\nrownames(mat) <- df$cell\nrequire(RColorBrewer)\npal = brewer.pal(100,"RdYlBu")\nheatmap.2(t(mat), trace="none",scale="row",col=pal, mar=c(3,12), main="Heatmap of Post-mapping QC metrics")')

