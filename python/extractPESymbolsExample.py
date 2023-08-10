get_ipython().run_line_magic('run', 'extractPESymbols.ipynb')

path = "/Users/ciou/Desktop/test/"
result = apiFrequencyCounting(path)
resultFile = open("APIList.txt", "w")
resultFile.write(" ".join( str(item) for item in result) + "\n")

