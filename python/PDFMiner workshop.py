from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage  

filename = 'MDOT_fastfacts02-2011_345554_7.pdf'
pagenums = [3] # empty list does all pages

output = StringIO()
manager = PDFResourceManager()
converter = TextConverter(manager, output, laparams=LAParams())
interpreter = PDFPageInterpreter(manager, converter)

with open(filename, 'rb') as fin:
    for page in PDFPage.get_pages(fin, pagenums):
        interpreter.process_page(page)

text = output.getvalue()
converter.close()
output.close()

text

from pprint import pprint as prettyprint
prettyprint(text)

savefile = filename.replace('pdf','txt')
with open(savefile,'w') as fout:
    fout.write(text)



