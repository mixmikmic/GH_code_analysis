import PyPDF2
import os
import string

# Modify this directory
pdf_dir = r"C:\Test"

# Iterate over each file in the directory
for filename in os.listdir(pdf_dir):
    full_name = os.path.join(pdf_dir, filename)
    # Open each file and read it using PyPDF2
    f = open(full_name, "rb")
    pdf = PyPDF2.PdfFileReader(f)
    title = pdf.getDocumentInfo().title + '.pdf'
    #title = pdf.getOutlines()[0].title + '.pdf'
    f.close()
    # Only allow valid characters in the string   
    valid = "-_.() %s%s" % (string.ascii_letters, string.digits)
    new_filename = ''.join(c for c in title if c in valid)   
    # Make sure the filename is unique
    if os.path.exists(os.path.join(pdf_dir, new_filename)):
        base, ext = os.path.splitext(new_filename)
        ii=1
        while os.path.exists(os.path.join(pdf_dir,base + "_" + str(ii) + ext)):
            ii+=1
        new_filename = base + "_" + str(ii) + ext
    # Rename the file
    full_new = os.path.join(pdf_dir, new_filename)
    os.rename(full_name, full_new)

