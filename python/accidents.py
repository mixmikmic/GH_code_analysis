from IPython.display import IFrame
IFrame(src="https://embed.polleverywhere.com/free_text_polls/D5B05A34g3E3kq6?controls=none&short_poll=true", width="100%", height="700")

import math
d = 0.1 # mGy

def q(e):
    exponent = -pow(math.log(2*e), 2)/6
    return 5.0 + 17.0*math.exp(exponent)

print("dose equivalent = ", d*q(2), " [mSv]")

from pyne import data 
print("data.Bq_per_Ci= ", data.Bq_per_Ci)
print("data.Ci_per_Bq= ", data.Ci_per_Bq)

get_ipython().magic('pinfo data.inhale_dose')

print("data.inhale_dose('129I') =",data.inhale_dose('129I'))
print("data.inhale_dose('137Cs') =", data.inhale_dose('137Cs'))
print("data.inhale_dose('134Cs') =", data.inhale_dose('134Cs'))

get_ipython().magic('pinfo data.ingest_dose')

print("data.ingest_dose('129I') =",data.ingest_dose('129I'))
print("data.ingest_dose('137Cs') =", data.ingest_dose('137Cs'))
print("data.ingest_dose('134Cs') =", data.ingest_dose('134Cs'))



# If we talk about this in Ci

print("TMI ",20, "Ci")
print("Fukushima ", (100E15)*data.Ci_per_Bq, "Ci")
print("Chernobyl ", (1760E15)*data.Ci_per_Bq, "Ci")

from pyne import pyne
data.dose_lung_model
data.dose_lung_model('137Cs')
s_per_d = pyne.utils.time_conv_dict['day']

isos = ['137Cs', '134Cs', '131I','129I']
for i in isos:
    print(i,
         data.half_life(i)/s_per_d,
         data.decay_children(i),
         data.ext_soil_dose(i),
         data.ext_air_dose(i))
    



