import gammalib
import ctools
import cscripts 
import csv

obs_num = 4

csv_filename = 'obs_test.csv'
name = 'Crab'
ra = 83.6331
dec = 22.0145
duration = 1800        # seconds
emin = 0.05          # in TeV
emax = 20.0          # in TeV
rad = 8.0
caldb = 'prod2'
irf = 'South_0.5h'
deadc = 0.95

#for i in range(obs_num):
#    offrad = 0.5*(i/4)
#    angle = (90*i)%360
#    ra_delta = offrad*gammalib.sind(angle)
#    dec_delta = offrad*gammalib.cosd(angle)


with open(csv_filename, 'w') as csvfile:
    fieldnames = ['name', 'id','ra','dec','duration', 'emin', 'emax','']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for id in range(1,obs_num+1):
        #routine to create some random pointings (0.5° offset step )
        # 4 pointings per each shell
        offrad = 1.0*(((id-1)/4)+1)
        angle = ((90)*(id-1))%360
        #ra_delta = offrad*gammalib.sind(angle)
        #dec_delta = offrad*gammalib.cosd(angle)
        ra_delta = 1.0
        dec_delta = 1.0
        
        id_print =str(id).rjust(4,'0')         # this create a 4-digits string filling with zeros
        writer.writerow({'name': name, 'id': id_print,'ra' : ra+ra_delta, 'dec':dec+dec_delta,'duration':duration, 'emin':emin, 'emax':emax})

# this is the csv file: see http://cta.irap.omp.eu/ctools/users/reference_manual/csobsdef.html for details.
get_ipython().system('more obs_test.csv')

# the script is called in this way because it was explicitely imported at the beginning

obsdef = cscripts.csobsdef()       #create empty instance
obsdef['inpnt'] = csv_filename     #Pointing definition ASCII file in comma-separated value (CSV) format.
obsdef['deadc'] = deadc
obsdef['caldb'] = caldb
obsdef['irf'] = irf
obsdef['rad'] = rad
obsdef['outobs'] = 'out_csobsdef.xml'
obsdef.execute()

#visualization of the Observation Definition XML file.
get_ipython().system("xmllint 'out_csobsdef.xml'")

#!xmllint $CTOOLS/share/models/crab.xml

