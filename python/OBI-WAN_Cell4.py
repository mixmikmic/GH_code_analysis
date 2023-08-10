get_ipython().system('docker run poldracklab/mriqc:latest --version')

get_ipython().system('docker run --rm -v c:/Users/kwie508/OBI-Neurotools-Project/data:/data:ro -v c:/Users/kwie508/OBI-Neurotools-Project/mriqc:/out poldracklab/mriqc:latest /data /out participant')

