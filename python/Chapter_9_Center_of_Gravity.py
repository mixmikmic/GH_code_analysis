# Example 9.1
from scipy import integrate
import math

# Calculation
a = lambda y: y**(2)*math.sqrt(4*y**(2)+1)
b = lambda y: math.sqrt(4*y**(2)+1)
xbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]

c = lambda y: y*math.sqrt(4*y**(2)+1)
d = lambda y: math.sqrt(4*y**(2)+1)
ybar = integrate.quad(c, 0, 1)[0]/integrate.quad(d, 0, 1)[0]  #[meter]

# Result
print"xbar = ",round(xbar,3),"m"
print"ybar = ",round(ybar,3),"m"

# Example 9.5
from scipy import integrate
import math
from __future__ import division

# Calculation
# Solution 1
a = lambda x: x**(3)
b = lambda x: x**(2)
xbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]
c = lambda x: (x**(4))/2
d = lambda x: x**(2)
ybar = integrate.quad(c, 0, 1)[0]/integrate.quad(d, 0, 1)[0]  #[meter]

# Result Solution 1
print"Solution 1"
print"xbar = ",round(xbar,3),"m"
print"ybar = ",round(ybar,3),"m\n"

# Solution 2
a = lambda y: (1-y)/2
b = lambda y: 1-math.sqrt(y)
xbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]
c = lambda y: y-y**(3/2)
d = lambda y: 1-math.sqrt(y)
ybar = integrate.quad(c, 0, 1)[0]/integrate.quad(d, 0, 1)[0]  #[meter]

# Result Solution 2
print"Solution 2"
print"xbar = ",round(xbar,3),"m"
print"ybar = ",round(ybar,3),"m"

# Example 9.6
from scipy import integrate
import math
from __future__ import division

# Calculation
# Solution 1
a = lambda x: x*(x-x**(2))
b = lambda x: (x-x**(2))
xbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]


# Result Solution 1
print"Solution 1"
print"xbar = ",round(xbar,3),"m\n"


# Solution 2
a = lambda y: ((math.sqrt(y)+y)/2)*(math.sqrt(y)-y)
b = lambda y: math.sqrt(y)-y
xbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]


# Result Solution 2
print"Solution 2"
print"xbar = ",round(xbar,3),"m"

# Example 9.7
from scipy import integrate
import math

# Calculation
a = lambda y: 100*math.pi*y**(2)
b = lambda y: 100*math.pi*y
ybar = integrate.quad(a, 0, 100)[0]/integrate.quad(b, 0, 100)[0]  #[millimeter]

# Result
print"ybar = ",round(ybar,1),"mm"

# Example 9.8
from scipy import integrate
import math

# Calculation
a = lambda z: z*200*z*math.pi*0.5**(2)
b = lambda z: 200*z*math.pi*0.5**(2)
zbar = integrate.quad(a, 0, 1)[0]/integrate.quad(b, 0, 1)[0]  #[meter]

# Result
print"zbar = ",round(zbar,3),"m"

# Example 9.9
import math

# Calculation
xbar = (60*math.pi*60+0*40+0*20)/(math.pi*60+40+20)  #[millimeter]
ybar = (-38.2*math.pi*60+20*40+40*20)/(math.pi*60+40+20)  #[millimeter]
zbar = (0*math.pi*60+0*40+-10*20)/(math.pi*60+40+20)  #[millimeter]

# Result
print"xbar = ",round(xbar,1),"mm"
print"ybar = ",round(ybar,1),"mm"
print"zbar = ",round(zbar,3),"mm"

# Example 9.10

# Calculation
xbar = (1*0.5*3*3+(-1.5)*3*3+(-2.5)*(-2)*1)/(0.5*3*3+3*3+(-2)*1)  #[meter]
ybar = (1*0.5*3*3+1.5*3*3+2*(-2)*1)/(0.5*3*3+3*3+(-2)*1)  #[meter]

# Result
print"xbar = ",round(xbar,3),"m"
print"ybar = ",round(ybar,2),"m"

# Example 9.11
import math
from __future__ import division

# Variable Declaration
pc = 8  #[milligram per meter cube]
ph = 4  #[milligram per meter cube]

# Calculation
zbar = (50*pc*10**(-6)*(1/3)*math.pi*50**(2)*200 + (-18.75)*ph*10**(-6)*(2/3)*math.pi*50**(3) + 125*(-pc)*10**(-6)*(1/3)*math.pi*25**(2)*100 + 50*(-pc)*10**(-6)*math.pi*25**(2)*100)/(pc*10**(-6)*(1/3)*math.pi*50**(2)*200+ph*10**(-6)*(2/3)*math.pi*50**(3)+(-pc)*10**(-6)*(1/3)*math.pi*25**(2)*100+(-pc)*10**(-6)*math.pi*25**(2)*100)  #[millimeter]

# Result
print"zbar = ",round(zbar,1),"mm"

# Example 9.13
from __future__ import division

# Variable Declaration
b = 1.5  #[meter]
pw = 1000  #[kilogram per meter cube]
# Calculation
# Solution 1
# Let water pressure at depth A be pA and water pressure at depth B be pB
pA = pw*9.81*2/1000  #[kilo Pascal]
pB = pw*9.81*5/1000  #[kilo Pascal]
wA = round(b*pA,2)  #[kilo Newton per meter]
wB = round(b*pB,2)  #[kilo Newton per meter]
# let FR be area of trapezoid
FR = round((1/2)*3*(wA+wB),1)  #[kilo Newton]
# Let h be force acting through centroid
h = round((1/3)*((2*wA+wB)/(wA+wB))*3,1)  #[meter]

# Result Solution 1
print"Solutuon 1"
print"FR = ",(FR),"kN"
print"h = ",(h),"m\n"

# Solution 2
FRe = round(wA*3,1)  #[kilo Newton]
Ft = round((1/2)*(wB-wA)*3,1)  #[kilo Newton]
FR = FRe + Ft  #[kilo Newton]
# +ΣMRB(clockwise) = ΣMB
h = round((FRe*1.5+Ft*1)/FR,1)  #[meter]

# Result Solution 2
print"Solutuon 2"
print"FR = ",(FR),"kN" 
print"h = ",(h),"m"

# Example 9.14
from __future__ import division

# Variable Declaration
b = 5  #[meter]
pw = 1020  #[kilogram per meter cube]

# Calculation
pB = round(pw*9.81*3/1000,2)  #[kilo Pascal]
wB = round(b*pB,1)  #[kilo Newton per meter]
F_x = round((1/2)*3*wB,1)  #[kilo Newton]
F_y = round(pw*9.81*5*(1/3)*1*3/1000,1)  #[kilo Newton]
FR = round(math.sqrt(F_x**(2)+F_y**(2)),0)  #[kilo Newton]

# Result
print"FR = ",(FR),"kN"

# Example 9.15
from scipy import integrate
from __future__ import division

# Calculation
a = lambda z: 9810*(z-z**(2))
F = round(integrate.quad(a, 0, 1)[0],2)  #[Newton]

# Resultant passes through centroid of volume
xbar = 0
a = lambda z: (9810/1635)*(z**(2)-z**(3))
zbar = round(integrate.quad(a, 0, 1)[0],1)  #[meter]

# Result
print"F = ",(round(F/1000,2)),"kN"
print"xbar = ",(xbar)
print"zbar = ",(zbar),"m"



