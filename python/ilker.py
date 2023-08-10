from IPython.display import Image
Image(filename='images/ts.jpg')

import math
Prated=75e3 #W
Trated=250 #Nm
p=6 #6 pole machine
pp=p/2 #pole pairs
Vrated=400 #V
Irated=110 #A
Vrotorinit=8000 #cm3
X=(math.pi/(4*pp))*math.sqrt(pp)
# X=l/D
dummy=Vrotorinit/(2*X*math.pi)
Rrotor=math.pow(dummy,0.3333)
Rrotor=round(Rrotor)
l=2*X*Rrotor
l=round(l)
Vrotor=math.pi*Rrotor*Rrotor*l
print Vrotor
print Rrotor
print l

from IPython.display import Image
Image(filename='images/indvolt.jpg')

from IPython.display import Image
Image(filename='images/slot2.jpg')

from IPython.display import Image
Image(filename='images/winding.jpg')

from IPython.display import Image
Image(filename='images/motor.jpg')

from IPython.display import Image
Image(filename='images/model.jpg')

from IPython.display import Image
Image(filename='images/torque.jpg')

