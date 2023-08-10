# Example Number 1.1

# Part(a)
# Variable Declaration
a = 10                        # [micro Newton(mN)]
b = 5                         # [Giga Newton(GN)]

# Calculation
# We have to find c = a * b
c = 10*5                      # [micro Newton(mN)*Giga Newton(GN)]
c = (10*10**(-3))*(5*10**(9)) # [N**(2)]
c = (10*10**(-3))*(5*10**(9))*10**(-6) #[kN**(2)]

#Result
print"Part(a)"
print "(10 mN)(5 GN) = ",int(c),"kilo Newton square\n"

# Part(b)
# Variable Declaration
a = 100                      #[millimeter(mm)]
b = 0.5**(2)                 #[mega Newton square(MN**(2))]

# Calculation
# We have to find c = a * b
c = (100*10**(-3))*(0.25*10**(12))  #[m.N**(2)]
c = (100*10**(-3))*(0.25*10**(12))*10**(-9) #[Gm.N**(2)]

#Result
print"Part(b)"
print "(100 mm)(0.5 MN square) = ",int(c),"Gigameter Newton square\n"

# Part(c) (Correction in the question (50 MN cube)(500 Gg))
# Variable Declaration
a = 50                     #[mega newton cube((MN)**(3))]
b = 500                    #[gigagram(Gg)]

# Calculation
# We have to find c = a / b
c = 50*(10**(6))**3 / 500*10**(6)     #[N**(3)/kg]
c = (50*((10**(6))**3) / (500*10**(6)))*10**(-9) #[kN**(3)/kg]

#Result
print"Part(c)"
print "(50 MN cube)(500 Gg) = ",int(c),"Kilo Newton cube per kg"



