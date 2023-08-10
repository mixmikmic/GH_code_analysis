import databaker.framework

databaker.framework.DATABAKER_INPUT_FILE = 'example1.xls'

f = databaker.framework.getinputfilename()
print(f)

databaker.framework.loadxlstabs(f)

