import urllib.request
import re

base_url = "http://aomol.msa.maryland.gov/000001/000538/pdf/am538--"
output_path = "../download/"
extension =".pdf"

# Looping through document 1 - 10
for i in range(1,10):
    print ("opening page ", i)
    
    # Get a file-like object for one of the city directories on the Internet Archive
    f = urllib.request.urlopen( base_url + str(i) + extension )

    # Read from the object, storing the page's contents in 's'.
    data = f.read()
    f.close()
    
    # Set the name of the downloaded file.
    output_name = "am538--" + str(i) + extension
        
    # Make an output file in your PC
    target = open( output_path + output_name, 'wb' )
    target.write( data )
    target.close()





