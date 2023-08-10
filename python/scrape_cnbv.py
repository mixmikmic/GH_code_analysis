# PACKAGES
import sys 
sys.path.append('C:/Dropbox/PythonTools')
	# https://github.com/skhiggins/PythonTools/
from get_files import get_files

# File types to scrape
file_types = ['xls','xlsx'] 

# Check working directory
import os
print(os.getcwd())

# Folder to save files into
dest_folder = os.path.join("..", "data")
dest_folder

# URL to scrape
cnbv_url = "http://portafolioinfo.cnbv.gob.mx/PortafolioInformacion/"

print(file_types)
get_files(cnbv_url, file_types, folder = dest_folder, overwrite=False)

cnbv_url_instructions = "https://www.gob.mx/cnbv/acciones-y-programas/banca-multiple-y-banca-de-desarrollo"
file_types_instructions = "pdf"

get_files(cnbv_url_instructions, 
    file_types_instructions, 
    folder = dest_folder, 
    overwrite=False
)

