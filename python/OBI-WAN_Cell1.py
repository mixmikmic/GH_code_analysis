get_ipython().system('docker pull rnair07/bidskit:v1.0 ')

get_ipython().system('docker run -it -v /PATH_TO_YOUR_RAW_DICOM_FOLDER/:/mnt rnair07/bidskit:v1.0 --indir=/mnt/DICOM --outdir=/mnt/BIDS')

get_ipython().system('docker run -it -v /PATH_TO_YOUR_RAW_DICOM_FOLDER/:/mnt rnair07/bidskit:v1.0 --indir=/mnt/DICOM --outdir=/mnt/BIDS')

