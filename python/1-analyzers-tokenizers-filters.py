import metapy
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")

tok = metapy.analyzers.ICUTokenizer()
tok.set_content(doc.content())
[t for t in tok]

tok = metapy.analyzers.ICUTokenizer()
tok = metapy.analyzers.LowercaseFilter(tok)
tok.set_content(doc.content())
[t for t in tok]

ana = metapy.analyzers.load('config.toml')
ana.analyze(doc)

ana = metapy.analyzers.NGramWordAnalyzer(1, tok)
ana.analyze(doc)

ana = metapy.analyzers.NGramWordAnalyzer(3, tok)
ana.analyze(doc)

