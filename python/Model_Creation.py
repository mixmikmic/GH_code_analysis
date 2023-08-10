fs = open('first_model', 'w')         #clean it if it's not empty
#fs = open('first_model','a')          #only append without overwriting

crab = "Crab Point 0 83.6331 22.0145 0 0 0 PL 5.7e-16 2.48 0.3*TeV"

fs.write(crab + ' \n')

crab_bkg = "BKG_crab CTAIrf 0 0 0 0 0 0 PL 1.0 0.0 0.3*TeV"
fs.write(crab_bkg + ' \n')

fs.close()

get_ipython().system('python scriptModel_variable.py first_model verbose')

fs = open('first_model', 'w')

crab = "c_gauss RadGauss 1 83.6331 22.0145 0.20 0 0 PL 5.7e-16 2.48 0.3*TeV"
crab_bkg = "BKGpoly Polynom 0 1.0_-0.123917_0.9751791_-3.0584577_2.9089535_-1.3535372_0.3413752_-0.0449642_0.0024321 0 0 0 0 PL 61.8e-6 1.85 1.0*TeV"

fs.write(crab + ' \n')
fs.write(crab_bkg + ' \n')

fs.close()

get_ipython().system('python scriptModel_variable.py first_model verb')



