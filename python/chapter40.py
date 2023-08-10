#A DC 2-wire feeder supplies a constant load with a sending-end voltage of 220V.Calculate the saving in copper
#if this voltage is doubled with power transmitted remaining the same.
##################################################################################################################



#Given
V1 = 220.0
V2 = 440.0
##Let us assume the wire has##
#length -> length 
#area   -> area
#current density -> cd
#power  -> P
P = 10000.0   #assumption
length = 1000.0    #assumption 
cd = 5.0    #assumption
#The values are assumed as these terms cancel out while calculating percentage.
I1 = P/V1
area  = I1/cd
#Vol of Cu required for 220V ->vol1
vol1 = 2*area*length


I2 = P/V2
area  = I2/cd
#Vol of Cu required for 440V ->vol2
vol2 = 2*area*length

#percentage saving of copper is
per_cent = ((vol1-vol2)/vol1)*100
print 'percentage saving in copper is ',per_cent,'%.'

#A uniform 2-wire d.c distributor 200 metres long is loaded with 2 amperes/metre.Resistance of
#single wire is 0.3 ohm/kilometre.Calculate the maximum voltage drop if the distributor is fed
#(a)from one end (b)from both ends with equal voltages.
#################################################################################################

#Given
length = 200.0        #metres
#current per unit length is
cur = 2.0             #amp/metre
#resistance per unit length is
res = 0.3/1000        #ohm/metre

#total resistance is
R = res*length        #ohm
#total current is
I = cur*length        #amp
#Total drop over a distributor fed from one end is given by
drop1 = (1/2.0)*I*R        #volts
#Total drop over a distributor fed from both ends is given by
drop2 = (1/8.0)*I*R
print 'Maximum voltage drop from one end is = ',drop1,'V.'
print 'Maximum voltage drop from both end is = ',drop2,'V.'

#A 2-wire d.c distributor AB is 300 metres long.It is fed at point A.The various loads and
#their positions are given below.
#      At point       distance from A in metres     concentrated load in A
#         C                     40                             30
#         D                     100                            40                
#         E                     150                            100
#         F                     250                            50
#If the maximum permissible voltage drop is not to exceed 10V,find the cross-sectional
#area of the distributor.Take resistivity = 1.78*10^(-8) ohm-m.
###########################################################################################


#Given
resistivity = 1.78e-8     #ohm-m
drop_max    = 10.0          #V
#loads and their positions
I1 = 30.0    #A
l1 = 40.0    #m
I2 = 40.0    #A
l2 = 100.0   #m
I3 = 100.0   #A
l3 = 150.0   #m
I4 = 50   #A
l4 = 250   #m
#We know that R = resistivity*length/Area
#Also max drop = I1*R1 + I2*R2 + I3*R3 + I4*R4 , using this
area = 2*(I1*l1 + I2*l2 + I3*l3 + I4*l4)*resistivity/drop_max   #m^2
area = area*1000000    #cm^2                 
print 'Cross sectional area of distributor =',area,'cm^2'

#A 2-wire d.c distributor F1F2 1000 metres long is loaded as under:
#Distance from F1(in metres): 100    250    500    600    700    800    850    920
#Load in amperes:              20     80     50     70     40     30     10     15
#The feeding points F1 and F2 are maintained at the same potential.Find which point will have the
#minimum potential and what will be the drop at this point?Take the cross-section of the conductors
#as 0.35 cm^2 and specific resistance of copper as 1.764*10^(-6) ohm-cm
#####################################################################################################

#Given
import numpy as np
resistivity = 1.764e-8           #ohm-m
area = 0.35e-4                      #m^2 
#loads and their positions taking from F1
I1 = 20         #A
l1 = 100        #m
I2 = 80         #A
l2 = 150        #m
I3 = 50         #A
l3 = 250        #m
I4 = 70        #A
l4 = 100        #m
I5 = 40         #A
l5 = 100        #m
I6 = 30         #A
l6 = 50        #m
I7 = 10         #A
l7 = 70        #m
I8 = 15         #A
l8 = 80       #m 

#sum of loads from F1
load1 = I1*l1 + I2*(l1+l2) + I3*(l1+l2+l3)                          #A-m
load2 = I8*l8 + I7*(l7+l8) + I6*(l6+l7+l8) + I5*(l5+l6+l7+l8)       #A-m

#guessing the point of minimum potential
if load1>load2:
    load2_new = load2 + I4*(l4+l5+l6+l7+l8)
    if load2_new>load1:
        pivot = I4

#solving 2 equations simultaneously
# x + y = 70(pivot) & 47000(load1) + 600(l1+l2+l3)x = 20,700(load2) + 400(l5+l6+l7+l8)y)
line1 = l1+l2+l3+l4                   #m
line2 = l4+l5+l6+l7+l8                #m 

a = [[1,1],[line1,-line2]]
b = [pivot,load2-load1]
soln = np.linalg.solve(a,b)            #soln is array with its elements[x,y]
#drop at minimum potential per conductor (in A-m)
drop_m = load1 + soln[0]*line1   #A-m

#resistance per metre = resistivity/Area
res = resistivity/area     #ohm/m

#Hence, drop in voltage per conductor is
drop = drop_m*res          #V  
#drop due to both is
drop = drop*2              #V

print 'Hence drop at minimum potential where load is',pivot,'A is =',round(drop,2),'V.'

#The resistance of a cable is 0.1ohm per 1000 metre for both conductors.It is loaded as shown in Fig.40.14(a).
#Find the current supplied at A and at B.If both the ends are supplied at 200 V
##############################################################################################################

#Given
#resistance per metre
res = 0.1/1000         #ohm/m
#loads and their positions taking from A
I1 = 50.0      #A
l1 = 500.0     #m
I2 = 100.0     #A
l2 = 700.0     #m
I3 = 150.0     #A
l3 = 300.0    #m
l4 = 250.0     #m    

#Assuming I flows from A to B
# equation is  res*[500*i + 700(i-50) + 300(i-150) + 250(i-300)] = 0
current_i = (I1*l2+(I1+I2)*l3 + (I1+I2+I3)*l4)/(l1+l2+l3+l4)
current_AC = current_i
current_CD = current_i-I1
current_DE = current_CD-I2
current_EB = current_DE-I3
if current_EB<0:
    current_EB = -current_EB;
print 'The current entering at A is = ',round(current_i,1),'A '
print 'The current entering at B is = ',round(current_EB,1),'A.'    

#The resistance of two conductors of a 2-conductor distributor shown in Fig.39.15 is 0.1ohm per 1000m
#for both conductors.Find (a)the current supplied at A(b)the current supplied at B
#(c)the current in each section (d)the voltages at C,D and E.Both A and B are maintained at 200V.
######################################################################################################

#Given
#resistance per metre
res = 0.1/1000         #ohm/m
#loads and their positions taking from A
I1 = 50.0      #A
l1 = 500.0     #m
I2 = 100.0     #A
l2 = 700.0     #m
I3 = 150.0     #A
l3 = 300.0    #m
l4 = 250.0     #m    

#Assuming I flows from A to B
# equation is  res*[500*i + 700(i-50) + 300(i-150) + 250(i-300)] = 0
current_i = (I1*l2+(I1+I2)*l3 + (I1+I2+I3)*l4)/(l1+l2+l3+l4)
current_AC = current_i
current_CD = current_i-I1
current_DE = current_CD-I2
current_EB = current_DE-I3
print "Current supplied at A is = ",round(current_i,1),"A."
print "Current supplied at B is = ",round(current_EB,1),"A."
print "Current in AC is = ",round(current_i,1),"A."
print "Current in CD is = ",round(current_CD,1),"A."
print "Current in DE is = ",round(current_DE,1),"A."
print "Current in EB is = ",round(current_EB,1),"A."
#Drop in volts is (resistance/metre)*length*current
drop_AC = res*l1*current_AC         #V
drop_CD = res*l2*current_CD          #V 
drop_DE = res*l3*current_DE          #V
print "Drop over AC is = ",round(drop_AC,1),"V."
print "Drop over CD is = ",round(drop_CD,1),"V."
print "Drop over DE is = ",round(drop_DE,1),"V."

#Voltages at C,D,E are
volt_C = 200-drop_AC        #V
volt_D = volt_C-drop_CD     #V
volt_E = volt_D-drop_DE     #V
print 'Voltage at C is = ',round(volt_C,1),'V.'
print 'Voltage at D is =',round(volt_D,1),'V.'
print 'Voltage at E is = ',round(volt_E,1),'V.'

#A 200 m long distributor is fed from both ends A and B at the same voltage of 250V.The
#concentrated loads of 50,40,30 and 25 A are coming on the distributor at distances of 50,75,
#100 and 150 m respectively from end A.Determine the minimum potential and locate its positions.
#Also,determine the current in each section of the distributor.It is given that the resistance
#of the distributor is 0.08ohm per 100 metres for go and return.
##################################################################################################


#Given
#resistance per metre
res = 0.08/100           #ohm/m
V_A = 250.0                #V
V_B = 250.0                #V
#load currents and their positions
I_C  = 50.0   #A
I_D  = 40.0   #A
I_E  = 30.0   #A
I_F  = 25.0   #A
l_AC = 50.0        #m
l_CD = 75.0 - l_AC   #m
l_DE = 100.0 - l_CD - l_AC        #m
l_EF = 150.0 - l_DE - l_CD - l_AC #m
l_FB = 200.0-150.0
#Assuming I flows from A to B
# equation is  res*[50*i + 25(i-50) + 25(i-90) + 50(i-120)+50(i-145)] = 0
current_i = (l_CD*I_C + l_DE*(I_C+I_D)+l_EF*(I_C+I_D+I_E) + l_FB*(I_C+I_D+I_E+I_F))/200.0
current_AC = current_i
current_CD = current_i-I_C
current_DE = current_CD-I_D
current_EF = current_DE-I_E
current_FB = current_EF-I_F
#As from figure in the book , point D is at minimum potential
if (current_CD>0) & (current_DE<0):
    point = "D"
    
#drop in volts = resistance/metre*sum(length*current) 
drop_d = res*(l_AC*current_AC + l_CD*current_CD)
min_pot = V_A-drop_d
print "Therefore point of minimum potential is",point,"and minimum potential is = ",round(min_pot,1),"V."    

#Each conductor of a 2-core distributor,500 metres long has a cross-sectional area
#of 2 cm^2.The feeding point A is supplied at 255V and the feeding point B at
#250V and load currents of 120A and 160A are taken at points C and D which are
#150 metres and 350 metres respectively from the feeding point A.Calculate the
#voltage at each load.Specific Resistance of copper is 1.7*10^(-6) ohm-cm
##################################################################################

#Given
area = 2e-4             #m^2
resistivity = 1.7e-8    #ohm-m
#load currents and their positions
i_c = 120.0    #A
i_d = 160.0    #A
l_ac = 150.0   #m
l_cd = 200.0   #m
l_db = 150.0   #m
V_a = 255.0    #V
V_b = 250.0    #V
#Resistance = resistivity*length/Area
#It is a 2 core distributor.Therefore,resistance per metre is
res = 2*resistivity/area   #ohm/m
#drop over whole distributor is equal to
drop = V_a - V_b     #V
#Therefore equation of total drop can be written as
# resistivity*(150i+200(i-120)+150(i-280))=5
current_i = (drop/res + l_cd*i_c + l_db*(i_c+i_d))/(l_ac+l_cd+l_db)   #A
current_ac = current_i          #A
current_cd = current_ac-i_c     #A
current_db = current_cd-i_d     #A

#Voltage at C = 255-drop over AC
volt_c = V_a-res*l_ac*current_ac    #V
#Voltage at D = 250-drop over DB    
volt_d = V_b -res*l_db*abs(current_db) #V
print "Voltage at point C is = ",round(volt_c,2),"V."
print "Voltage at point D is = ",round(volt_d,2),"V."

#A 2-wire distributor 500 metres long is fed at P at 250V and loads of 40A,20A,60A,30A are tapped off
#off from points A,B,C and D which are at distances of 100 metres,150 metres,300 metres and 400 metres
#from P respectively.The distributor is also uniformly loaded at the rate of 0.1A/m.If the resistance of
#the distributor per metre(go and return) is 0.0005 ohm,calculate the voltage at(i)pointQ and(ii)point B.
###########################################################################################################

#Given
V_P = 250.0    #V
resistance = 0.0005        #ohm/m

#loads and their positions
i_a = 40.0     #A
i_b = 20.0     #A
i_c = 60.0     #A
i_d = 30.0     #A
l_pa = 100.0     #m
l_ab = 150.0-100.0      #m
l_bc = 300.0-150.0     #m
l_cd = 400.0-300.0     #m
#uniform dstributed load
cur_uni = 0.1      #A/m


#considering drop due to concentrated loading
drop_pa = (i_a+i_b+i_c+i_d)*l_pa*resistance      #V
drop_ab = (i_b+i_c+i_d)*l_ab*resistance          #V 
drop_bc = (i_c+i_d)*l_bc*resistance              #V
drop_cd = i_d*l_cd*resistance                    #V
tot_drop = drop_pa + drop_ab + drop_bc + drop_cd      #V

#considering drop due to uniform loading(drop = irl^2/2) l = 500m
drop_uni = cur_uni*resistance*(500.0*500.0)/2     #V

V_Q = V_P - (tot_drop + drop_uni)   #V
#for point B
#drop due to concentrated loading
drop_b = drop_pa + drop_ab        #V
#drop due to uniform loading (drop = ir(lx-x^2/2)) l=500m x=150m
drop_ub = cur_uni*resistance*(500*(l_pa+l_ab)-(l_pa+l_ab)*(l_pa+l_ab)/2)   #V

V_B = V_P - (drop_b + drop_ub)    #V

print "Volatge at point Q is = ",round(V_Q,2),"V."
print "Voltage at point B is = ",round(V_B,2),"V." 

#A distributor AB is fed from both ends.At feeding point A,the voltage is maintained at 236V and at B at 237V.
#The total length of the distributor is 200 metres and loads are tapped off as under:
#(i) 20A at 50 metres from A   (ii) 40A at 75 metres from A. (iii)25A at 100 metres from A (iv)30A at 150 metres from A
#The reistance per kilometre of one conductor is 0.4ohm.Calculate the currents in the various sections of the distributor,
#the minimum voltage and the point at which it occurs.
###########################################################################################################################


#Given
#resistance per metre
res = 2*0.4/1000      #ohm/m
V_a = 236.0    #V
V_b = 237.0    #V
#loads and their positions
i_c = 20.0  #A
i_d = 40.0  #A
i_e = 25.0  #A
i_f = 30.0  #A
l_ac = 50.0  #m
l_cd = 25.0  #m
l_de = 25.0  #m
l_ef = 50.0  #m
l_fb = 50.0  #m
#Voltage drop equation  res*(50i + 25(i-20)+25(i-60) + 50(i-85) + 50(i-115)=-1)
current_i = ((V_a-V_b)/res + l_cd*(i_c)+l_de*(i_c+i_d)+l_ef*(i_c+i_d+i_e)+l_fb*(i_c+i_d+i_e+i_f))/200.0
current_ac = current_i
current_cd = current_ac-i_c
current_de = current_cd-i_d
current_ef = current_de-i_e
current_fb= current_ef-i_f
if current_cd>0:
    if current_de<0:
        point = "D"
#Minimum potential is at D as shown in figure
drop = res*(current_ac*l_ac + current_cd*l_cd)
V_d = V_a-drop
print "Current in section AC is = ",round(current_ac,2),"A."
print "Current in section CD is = ",round(current_cd,2),"A."
print "Current in section DE is = ",round(current_de,2),"A."
print "Current in section EF is = ",round(current_ef,2),"A."
print "Current in section FB is = ",round(current_fb,2),"A."
print "Minimum voltage is at point",point,"and minimum voltage is = ",round(V_d,2),"V."

#A distributor cable AB is fed at its ends A and B.Loads of 12,24,72 and 48 A are taken from the cable at
#points C,D,E and F.The resistances of sections AC,CD,DE,EF and FB of the cable are 8,6,4,10 and 5 milliohm
#respecively(for the go and return conductors together). The potential difference at point A is 240V,the p.d
#at the load F is also to be 240V.Calculate the voltages at the feeding point B,the current supplied by each
#feeder and the p.d.s at the loads C,D and E.
##############################################################################################################

#Given
V_a = 240.0    #V  
V_f = 240.0    #V
#loads and their resistances.
i_c = 12.0  #A
i_d = 24.0  #A
i_e = 72.0  #A
i_f = 48.0  #A

r_ac = 8e-3   #ohm
r_cd = 6e-3   #ohm
r_de = 4e-3   #ohm
r_ef = 10e-3  #ohm
r_fb = 5e-3   #ohm

#Voltage drop accross AF is zero.Therefore equation 8i +6(i-12) + 4(i-36)+10(i-108)*10^(-3)
current_i = (r_cd*i_c + r_de*(i_c+i_d) + r_ef*(i_c+i_d+i_e))/(28.0e-3)   #A
#currents in different sections
current_ac = current_i        #A
current_cd= current_ac-i_c    #A
current_de = current_cd-i_d   #A
current_ef = current_de-i_e   #A   
current_fb = current_ef-i_f   #A
#voltage at different points are
V_b = V_f - current_fb*r_fb   #V
V_c = V_a - current_ac*r_ac   #V
V_d = V_c - current_cd*r_cd   #V
V_e = V_d - current_de*r_de   #V 

print "Current supplied by feeder at point A is",round(current_ac,2),"A and that by point B is",round(abs(current_fb),2),"A."
print "Voltage at point B is = ",round(V_b,2),"V."
print "Voltage at point C is = ",round(V_c,2),"V."
print "Voltage at point D is = ",round(V_d,2),"V."
print "Voltage at point E is = ",round(V_e,2),"V."

#A two-wire d.c sdistributor PQ,800 metre long is loaded as under:
#Distance from P(metres):    100      250      500      600      700
#Loads in amperes:            20       80       50       70       40
#The feeding point at P is maintained at 248V and that at Q at 245V.The total resistance of
#the distributor(lead and return) is 0.1 ohm.Find (a)the current supplied at P and Q and
#(b)the power dissipated in the distributor.
##################################################################################################

#Given
V_p  = 248.0     #V
V_q  = 245.0     #V
res  = 0.1/800    #ohm/m 
#loads and their positions
i1 = 20.0     #A
i2 = 80.0     #A
i3 = 50.0     #A
i4 = 70.0     #A
i5 = 40.0     #A
l1 = 100.0           #m
l2 = 250.0-100.0     #m
l3 = 500.0 -250.0    #m
l4 = 600.0-500.0     #m
l5 = 700.0-600.0     #m
l6 = 800.0-700.0     #m
#drop accross the distributor :- 1/8000(100i + 150(i-20) + 250(i-100)+ 100(i-150)+100(i-220)+100(i-260) )=3
current_i = ((V_p-V_q)/res + l2*i1+l3*(i1+i2)+l4*(i1+i2+i3)+l5*(i1+i2+i3+i4)+l6*(i1+i2+i3+i4+i5))/800.0
current_p = current_i       #A
current_2 = current_p-i1    #A
current_3 = current_2-i2    #A
current_4 = current_3-i3    #A
current_5 = current_4-i4    #A
current_q = current_5-i5    #A
#Power loss = sum(I^2R)
loss = res*(current_p*current_p*l1 + current_2*current_2*l2 + current_3*current_3*l3 + current_4*current_4*l4 + current_5*current_5*l5 + current_q*current_q*l6)
print "The current supplied at P is = ",round(current_p,2),"A."
print "The current supplied at Q is = ",round(abs(current_q),2),"A."
print "Power dissipated in distributor is =",round(loss,2),"W."

#The two conductors of a d.c distributor cable 1000m long have a total resistance of 0.1 ohm.
#The ends A and B are fed at 240V.The cable is uniformly loaded at 0.5 A per metre length
#and has concentrated loads of 120A,60A,100A and 40A at points distant 200,400,700 and 900m.
#respectively from the end A.Calculate (i)the point of minimum potential on the distributor
#(ii)the value of minimum potential and (iii) currents fed at the ends A and B.
###############################################################################################

#Given
V_a = 240.0   #V
V_b = 240.0   #V
res = 0.1/1000   #ohm/m
#concentrated loads and their positions
i_c = 120.0       #A
i_d = 60.0        #A
i_e = 100.0       #A
i_f = 40.0        #A
l_ac = 200.0      #m
l_cd = 400.0-200.0    #m
l_de = 700.0-400.0    #m
l_ef = 900.0-700.0    #m
l_fb = 1000.0-900.0    #m
#Uniform loading
cur_uni = 0.5     #A/m
#Equation for drop from A to B -> (1/10000)*(200i + 200(i-120)+ 300(i-180)+200(i-280)+100(i-320))=0
current_i = (l_cd*i_c + l_de*(i_c+i_d)+l_ef*(i_c+i_d+i_e)+l_fb*(i_c+i_d+i_e+i_f))/1000

#point of minimum potential
current_ac = current_i       #A
current_cd = current_ac-i_c    #A
current_de = current_cd-i_d    #A
current_ef = current_de-i_e    #A
current_fb = current_ef-i_f    #A

if current_cd>0:
    if current_de<0:
        point = "D"
#As from figure it is inferred that point of minimum potential is D.
#Therefore,uniform load from point A to D(supplied from A)
cur_uni_A = cur_uni*(l_ac + l_cd)    #A
cur_A = cur_uni_A + current_ac     #A
#Therefore,uniform load from point B to D(supplied from B)
cur_uni_B = cur_uni*(l_de + l_ef + l_fb)    #A
cur_B = cur_uni_B + abs(current_fb)     #A

#drop at D due to concentrated load(from A)
drop_con = res*(current_ac*l_ac + current_cd*l_cd)
#drop at D due to uniform load(from A)[formula-> irl^2/2]
drop_uni = cur_uni*res*(l_ac+l_cd)*(l_ac+l_cd)/2
#total drop is
drop_tot = drop_con + drop_uni

#potential at D is
V_d = V_a - drop_tot
print "The point of minimum potential is",point,"and minimum potential is = ",round(V_d,2),"V."
print "Current fed at the end A is = ",round(cur_A,2),"A."
print "Current fed at the end B is = ",round(cur_B,2),"A."

#It is proposed to lay out a d.c distribution system comprising three sections-the first section consists
#of a cable from the sub-station to a point distant 800 metres from which two cables are taken,one 350 metres
#long supplying a load of 22kW and the other 1.5 kilometre long and supplying a load of 44kW.Calculate the
#cross-sectional area of each cable so that the total weight of copper required shall be minimum if the maximum
#drop of voltage along the cable is not to exceed 5% of the normal voltage of 440V at the consumer's premises.
#Take specific resistance of copper at working temperature equal to 2*10e-7 ohm-cm.
###################################################################################################################

#Given
resistivity = 2*10e-7   #ohm-cm
dist = 800.0*100          #cm
#Current taken from 350m section
cur_1 = 22000.0/440
#Current taken from 1.5km section
cur_2 = 44000.0/440
#Therefore,current in first section
cur = cur_1 + cur_2
#Let us assume
#V->voltage drop accross first section
#R->resistance of the first section
#A->cross-sectional area of te first section

from sympy import Eq, var, solve
var('V') 
#Now , R = V/I
R = V/cur
# A = resistivity*l/R -> A = resistivity*l*I/V 
A = resistivity*dist/R
#Max allowable drop
max_drop = (5.0/100)*440.0
#Voltage drop along other sections
vol_drop = max_drop - V
#Cross-sectional area of 350 m   A = resistivity*l/R 
A1 = resistivity*350.0*100*cur_1/(vol_drop)
#Cross-sectional area of 1.5km   A = resistivity*l/R 
A2 = resistivity*1500.0*100*cur_2/(vol_drop)


#Now,Total weight is propotional to total volume 
W = 800.0*A + 350.0*A1+1500.0*A2
Diff = W.diff(V)
eq = Eq(Diff,0)

V = solve(eq)
#We get 2 values of V of which Negative is not possible.Therefore,
V = float(V[1])
A = resistivity*dist*cur/V
vol_drop = max_drop - V
A1 = resistivity*350.0*100*cur_1/vol_drop
A2 = resistivity*1500.0*100*cur_2/vol_drop
print "Voltage V is = ",round(V,2),"V."
print "Cross-sectional Area A is = ",round(A,2),"cm^2."
print "Cross-sectional Area A1 is = ",round(A1,2),"cm^2."
print "Cross-sectional Area A2 is = ",round(A2,2),"cm^2."



#A d.c two-wire distributor AB is 450m long and is fed at both ends at 250 volts.It is loaded as follows:20A at 60m from A,
#40A at 100m from A and a uniform loading of 1.5A/m from 200 to 450m from A.The resistance of each conductor is
#0.05ohm/km.Find the point of minimum potential and its potential.
####################################################################################################################

#Given
V_a = 250.0  #V
V_b = 250.0  #V
res = 0.05/1000    #ohm/m
cur_uni = 1.5      #A/m (uniform loading)
#loads and positions
i_c = 20.0   #A
i_d = 40.0   #A
l_ac = 60.0   #m
l_cd = 40.0   #m
l_de = 100.0  #m
l_eb = 250.0  #m

#Let us assume that point of minimum potential is D and let i be current in section CD.
#Therefore,current from B is (40-i).If r is resistance then
#(20+i)*60r + i*40r = (40-i)*350r + 1.5*r*250^2/2   [drop over AD = drop over BD as V_a = V_b]

cur_i = (i_d*(l_de+l_eb)*res + cur_uni*res*l_eb*l_eb/2 - i_c*l_ac*res)/((l_ac+l_cd+l_de+l_eb)*res)   #A

#cur_i > 40 i.e 40-i is negative,it means D is not point of minimum potential.Let F be point of minimum potential(between DB)
#current in section DF is
cur_df = cur_i-i_d      #A

#distance EF
dist_ef = cur_df/cur_uni    #m

#distance of F from A is
dist = l_ac + l_cd + l_de + dist_ef   #m

#total drop over AF is [(20+i)*60r + i*40r+ (i-40)*161.7r - 1.5*r*61.7^2/2
drop_af = 2*res*((i_c+cur_i)*l_ac + cur_i*l_cd + cur_df*(l_de+dist_ef)-cur_uni*dist_ef*dist_ef/2)  #V
#potential at F
V_f = V_a - drop_af      #V
print "The point of minimum potential is at",round(dist,2),"m from A."
print "The minimum potential is = ",round(V_f,2),"V."

#A two-wire d.c distributor AB,1000 metres long,is supplied from both ends,240V at A and 242V at B.There is a
#concentrated load of 200A at a distance of 400 metre from A and a uniformly distrubuted load of 1.0A/m between
#the mid-point and end B.Determine (i)the currents fed at A and B(ii)the point of minimum potential and
#(iii)voltage at this point.Take cable resistance as 0.005 ohm per 100 metre each core.
#####################################################################################################################

#Given
#resistance per 100 metres
res = 2*0.005/100         #ohm/m
cur_uni = 1.0           #A/m
cur_con = 200.0         #A
len_uni = 500.0
#Let us assume that Ib current flows from point B.
#Considering a element dx in BD(500 metres) at a distance of X units(100 m each)
#voltage drop over dx = (1-100*x)*res*dx
#voltage drop over BD by integrating is = 0.05*Ib - 12.5
#voltage drop over DC = (Ib-500)*0.01
#voltage drop over CA = (Ib-700)*0.01*4
#total drop over AB = 
tot_drop = 242.0-240.0
#summation of drops from AC + CD + DB
from sympy import Eq, var, solve
var('Ib') 
sum =  (Ib-500)*0.01 +(Ib-700)*0.01*4 + 0.05*Ib - 12.5

eq = Eq(sum,tot_drop)

Ib =  solve(eq)
Ib = float(Ib[0])
#Total current
cur_tot = len_uni*cur_uni + cur_con
Ia = cur_tot - Ib                 #A
#Current in distributed load
cur_dis = Ia-cur_con              #A
#point of minimum potential from D is
distD = cur_dis/cur_uni
#Therefore distance from B is
distB = len_uni-distD
#Therefore voltage drop is
from scipy.integrate import quad

def integrand(x):
    return (Ib-100*x)*res*100

ans, err = quad(integrand, 0, (distB/100))
#Therefore potential of M is
pot_M = 242.0-ans     #V
print "Current fed at A is = ",Ia,"A."
print "Current fed at B is = ",Ib,"A."
print "Point of minimum potential from B is = ",distB,"metres."
print "Voltage at minimum potential is = ",round(pot_M,2),"V."

#A 400-metre ring distributor has loads as shown in Fig. 40.29(a) where distances are in metres.The resistance
#of each conductor is 0.2 ohm per 1000 metres and the loads tapped off at points B,C,D are as shown.If the
#distributor is fed at A,find voltages at B,C and D.
#################################################################################################################

#Given

res = 0.2/1000   #ohm/m
V_a = 240.0       #V
#loads and positions
i_b = 100.0  #A
i_c = 70.0   #A
i_d = 50.0   #A
l_ab = 60.0   #m
l_bc = 80.0   #m
l_cd = 90.0   #m
l_da = 70.0   #m

#total drop ->70i + 90(i-50)+80(i-120)+60(i-220)=0
cur_i = (l_cd*i_d + l_bc*(i_d+i_c) + l_ab*(i_d+i_c+i_b))/(l_ab+l_bc+l_cd+l_da)
#drops in different sections
drop_da = 2*cur_i*l_da*res
drop_cd = 2*(cur_i-i_d)*l_cd*res
drop_bc = 2*abs(cur_i-i_d-i_c)*l_bc*res
drop_ab = 2*abs(cur_i-i_d-i_c-i_b)*l_ab*res

#voltages at different points
V_d = V_a - drop_da
V_c = V_d - drop_cd
V_b = V_a - drop_ab
print "Voltage at B is = ",round(V_b,2),"V."
print "Voltage at C is = ",round(V_c,2),"V."
print "Voltage at D is = ",round(V_d,2),"V."

#In a direct current ring main,a voltage of 400V is maintained at A.At B,500 metres away from A,a load of 150A is taken
#and at C,300 metres from B,a load of 200A is taken.The distance between A and C is 700 metres.The resistance of each
#conductor of the mains is 0.03ohm per 1000 metres.Find the voltage at B and C and also find the current in the section BC.
##############################################################################################################################

#Given
V_a = 400.0    #V
res = 0.03/1000   #ohm/m
#loads and positions
i_b = 150.0    #A
i_c = 200.0    #A
l_ab = 500.0   #m
l_bc = 300.0   #m
l_ca = 700.0   #m

#total drop-> 500i + 300(i-150) + 700(i-350) = 0
cur_i = (l_bc*i_b + l_ca*(i_b+i_c))/(l_ab+l_bc+l_ca)
#current in different sections
cur_ab = cur_i
cur_bc = cur_i-i_b
cur_ca = abs(cur_bc-i_c)
#drops in different sections
drop_ab = 2*cur_ab*l_ab*res
drop_bc = 2*cur_bc*l_bc*res
#voltages in different sections
V_b = V_a-drop_ab
V_c = V_b-drop_bc
print "Voltage at B is = ",round(V_b,2),"V."
print "Voltage at C is = ",round(V_c,2),"V."
print "Current in section BC is = ",round(cur_bc,2),"A."

#A d.c ring main ABCDE is fed at point A from a 220-V supply and the resistances(including both lead and return)
#of the various sections are as follows(in ohms):AB=0.1;BC=0.05;CD=0.01;DE=0.025 and EA=0.075.The main supplies
#loads of 10A at B; 20A at C; 30A at D and 10A at E.Find the magnitude and direction of the current flowing in each
#section and the voltage at each load point.
#If the points C and E are further linked together by a conductor of 0.05 ohm resistance and the output currents
#from the mains remain unchanged,find the new distribution of the current and voltage in the network.
#####################################################################################################################

#Given

V_a = 220.0      #V
#resistances of different sections
r_ab = 0.1      #ohm
r_bc = 0.05     #ohm
r_cd = 0.01     #ohm
r_de = 0.025    #ohm
r_ea = 0.075    #ohm
#loads
i_b = 10.0    #A
i_c = 20.0    #A
i_d = 30.0    #A
i_e = 10.0    #A
#total drop -> 0.1i + 0.05(i-10) + 0.01(i-30) + 0.025(i-60) + 0.075(i-70)=0
cur_i = (r_bc*i_b + r_cd*(i_b+i_c) + r_de*(i_b+i_c+i_d) + r_ea*(i_b+i_c+i_d+i_e))/(r_ab+r_bc+r_cd+r_de+r_ea)
#current in different sections
cur_ab = cur_i
cur_bc = cur_ab-i_b
cur_cd = cur_bc-i_c
cur_de = cur_cd-i_d
cur_ea = cur_de-i_e

#drops in different sections
drop_ab = cur_ab*r_ab
drop_bc = cur_bc*r_bc
drop_de = abs(cur_de)*r_de
drop_ea = abs(cur_ea)*r_ea
#voltages at different points
V_b = V_a - drop_ab
V_c = V_b - drop_bc
V_e = V_a - drop_ea
V_d = V_e - drop_de
print "Current in AB,BC,CD,DE,EA is",round(cur_ab,2),"A,",round(cur_bc,2),"A,",round(abs(cur_cd),2),"A,",round(abs(cur_de),2),"A,",round(abs(cur_ea),2),"A respectively." 
print ""
print "Voltage at B,C,D,E is",round(V_b,2),"V,",round(V_c,2),"V,",round(V_d,2),"V,",round(V_e,2),"V respectively"
print ""
#part-2
#Potential difference between end points of interconnector(CE)
V_ce = V_e-V_c
#Resistance between CE ,as shown in figure
r1 = r_ab+r_bc+r_ea
r2 = r_de + r_cd
res_ce = r1*r2/(r1+r2)+ 0.05

#Current in interconnector [I = V/R Ohm's Law]
cur_ce = V_ce/res_ce
#Current goes from E to C as E is at higher potential.

#The current in other sections will also change.
#let us assume i1 along ED, voltage round the closed mesh EDC is zero.
#total drop -> -0.025*i1-0.01*(i1-30)+0.05*9.75 = 0

cur_i1 = (0.05*cur_ce + r_cd*i_d)/(r_cd+r_de)

current_ea = i_e+cur_i1+cur_ce
current_ab = (i_b+i_c+i_d+i_e)-current_ea
current_bc = current_ab-i_b
current_de = current_ea-i_e
#new drops
drop_ab = current_ab*r_ab
drop_bc = current_bc*r_bc
drop_ea = current_ea*r_ea
drop_de = current_de*r_de

#new potentials
V_b = V_a - drop_ab
V_c = V_b - drop_bc
V_e = V_a - drop_ea
V_d = V_e - drop_de

print "Current in AB,BC,DE,CE,EA is",round(current_ab,2),"A,",round(current_bc,2),"A,",round(current_de,2),"A,",round(cur_ce,2),"A,",round(current_ea,2),"A respectively."
print ""
print "Voltage at B,C,D,E is",round(V_b,2),"V,",round(V_c,2),"V,",round(V_d,2),"V,",round(V_e,2),"V respectively"                                 

#In a 3-wire distribution system,the supply voltage is 250V on each side.The load on one side is a 3 ohm
#resistance and on the other, a 4 ohm resistance.The resistance of each of the 3 conductors is 0.05 ohm.
#Find the load voltages.
#########################################################################################################

import numpy as np
#Given
#Resistances
res_1 = 3.0   #ohm
res_2 = 4.0   #ohm
res_con = 0.05 #ohm
V_sup = 250.0  #V

#Let the assumed directions of unknown currents be as shown in figure.
#KVL for ABCD
# (3+0.05)x + 0.05(x-y) = 250  -------------- eqn 1
a = res_1 + 2*res_con
b = -(res_con)
#KVL for DCEFD
# 0.05(y-x) + (4+0.05)y = 250  -------------- eqn 2
c = res_2+ 2*res_con       
#Solving eqn 1 and eqn2
m = [[a,b],[b,c]]
n = [V_sup,V_sup]
soln = np.linalg.solve(m,n)            #soln is array with its elements[x,y]
#Calculating the load voltages
#V1 = 250-0.05*x-0.05(x-y)
vol1 = V_sup - res_con*soln[0]-res_con*(soln[0]-soln[1])    #V
#V2 = 250 + 0.05(x-y)- 0.05y
vol2 = V_sup + res_con*(soln[0]-soln[1]) - res_con*soln[1]  #V
print "Voltage across 3 ohm load is = ",round(vol1,1),"V."
print "Voltage across 4 ohm load is = ",round(vol2,1),"V."

#A 3-wire d.c distributor PQ,250 metres long,is supplied at end P at 500/250V and is loaded as under:
#Positive side: 20A 150 metres from P ; 30A 250 metres from P.
#Negative side: 24A 100 metres from P ; 36A 220 metres from P.
#The resistance of each outer wire is 0.02 ohm per 100 metres and the cross-section of the middle wire
#is one-half of the outer.Find the voltage across each load point.
##########################################################################################################

#Given
V_PN = 250.0 #V
V_NR = 250.0 #V
res_out = 0.02/100      #ohm/m
res_mid = 2*res_out       #ohm/m  (Area of middle wire is half.As, R = rho*l/A .Therefore,Resistance doubles)

#Given Currents
i_ab = 20.0  #A
i_qk = 30.0  #A
i_cd = 24.0  #A
i_fe = 36.0  #A

#Currents in different sections
i_pa = i_ab+i_qk   #A
i_aq = i_qk        #A
i_fk = i_qk        #A
i_bf = i_fe-i_qk   #A
i_bc = i_ab-i_bf   #A
i_cn = i_cd-i_bc   #A
i_de = i_fe        #A
i_dr = i_cd+i_fe   #A


#lengths of different sections
l_pa = 150.0 #m
l_aq = 100.0 #m
l_kf = 250.0-220.0 #m
l_bc = 150.0-100.0 #m
l_bf = 220.0-150.0 #m
l_cn = 100.0 #m
l_de = 220.0-100.0 #m
l_dr = 100.0  #m

#Resistances of different sections
r_pa = l_pa*res_out #ohm
r_aq = l_aq*res_out #ohm
r_kf = l_kf*res_mid #ohm
r_bc = l_bc*res_mid #ohm
r_bf = l_bf*res_mid #ohm
r_cn = l_cn*res_mid #ohm
r_de = l_de*res_out #ohm
r_dr = l_dr*res_out #ohm

#Drop across different sections
drop_pa = r_pa*i_pa #V
drop_aq = r_aq*i_aq #V
drop_kf = r_kf*i_fk #V
drop_bc = r_bc*i_bc #V
drop_bf = r_bf*i_bf #V
drop_cn = r_cn*i_cn #V
drop_de = r_de*i_de #V
drop_dr = r_dr*i_dr #V

#Voltages across different sections
vol_ab = V_PN - drop_pa - drop_bc + drop_cn       #V
vol_qk = vol_ab - drop_aq - drop_kf + drop_bf     #V
vol_cd = V_NR - drop_cn - drop_dr                 #V  
vol_fe = vol_cd + drop_bc - drop_bf - drop_de     #V

print "Potential Difference across AB is = ",round(vol_ab,2),"V."
print "Potential Difference across QK is = ",round(vol_qk,2),"V."
print "Potential Difference across CD is = ",round(vol_cd,2),"V."
print "Potential Difference across FE is = ",round(vol_fe,2),"V."

#A d.c 3-wire system with 500-V between outers has lighting load of 100kW on the positive and 50kW on the
#negative side.If,at this loading,the balancer machines have each a loss of 2.5kW,Calculate the kW loading
#of each balancer machine and the total load on the system.
###########################################################################################################

#Given
V_out = 500.0  #V
load_p = 100.0 #kW (positive side)
load_n = 50.0  #KW (negative side)
load_b = 2.5   #kW  (balancer machine)
#total load on main generator
load_tot = load_p + load_n + 2*load_b  #kW
#Output current of main generator
cur_out = load_tot*1000/V_out   #W/V->A
#load current on positive side
cur_p = load_p*1000/(V_out/2)  #A
#load current on negative side
cur_n = load_n*1000/(V_out/2)  #A
#Current through neutral(Out of balance)
cur_o = cur_p-cur_n     #A

#Currents of balancer
cur_b1 = cur_p-cur_out  #A
cur_b2 = cur_o - cur_b1 #A

#Load on balancer
load_b1 = (V_out/2)*cur_b1/1000    #kW
load_b2 = (V_out/2)*cur_b2/1000    #kW

print "Total load on main generator is = ",round(load_tot,2),"kW."
print "Load on Balancer 1 is = ",round(load_b1,2),"kW."
print "Load on Balancer 2 is = ",round(load_b2,2),"kW."

#In a 500/250-V d.c 3-wire system,there is a current of 2000A on the +ve side, 1600A on the negative side
#and a load of 300 kW across the outers.The loss in each balancer set is 8 kW.Calculate the current in each
#armature of the balancer set and total load on the main generator.
#############################################################################################################

#Given
V_out    = 500.0  #V
cur_p    = 2000.0 #A (current on positive side)
cur_n    = 1600.0 #A (current on negative side)
load_ext = 300.0 #kW (across outers)
load_b   = 8.0   #kW  (loss in balancer set)
#loading on positive side
load_p = (cur_p*(V_out/2))/1000  #kW
#loading on negative side
load_n = (cur_n*(V_out/2))/1000  #kW
#Total loading on main generator
load_tot = load_p + load_n + 2*load_b + load_ext   #kW

#current on main generator -> I = W/V
cur_tot = load_tot*1000/V_out     #A

#current through neutral(out of balance)
cur_o = cur_p-cur_n    #A

#current through external resistance
cur_ext = load_ext*1000/V_out    #A

#current through balancer sets
cur_b1 = (cur_p+cur_ext)-cur_tot      #A
cur_b2 = cur_o - cur_b1               #A

print "Total load on main generator is = ",round(load_tot,2),"kW."
print "Current through Balancer 1 is = ",round(cur_b1,2),"A."
print "Current through Balancer 2 is = ",round(cur_b2,2),"A."

#On a 3-wire d.c distribution system with 500V between outers,there is a load of 1500kW on the positive
#side and 2000 kW on the negative side.Calculate the current in the neutral and in each of the balancer
#armatures and the total current supplied by the generator.Neglect losses.
##########################################################################################################

#Given
V_out = 500.0    #V
load_p = 1500.0  #kW   (load on positive side)
load_n = 2000.0  #kW   (load on negative side)
#total loading on main generator
load_tot = load_p + load_n    #kW
#current supplied by generator
cur_tot = load_tot*1000/V_out    #A
#current on positive side
cur_p = load_p*1000/(V_out/2)    #A
#current on negative side
cur_n = load_n*1000/(V_out/2)    #A
#current in neutral(out of balance)
cur_o = abs(cur_p-cur_n)   #A
#current through armatures
cur_b1 = cur_tot-cur_p  #A
cur_b2 = cur_o-cur_b1   #A

print "Current supplied by generator is = ",cur_tot,"A."
print "Current in positive side is = ",cur_p,"A."
print "Current in negative side is = ",cur_n,"A."
print "Current in neutral is = ",cur_o,"A."
print "Current through armature 1 is = ",cur_b1,"A."
print "Current through armature 2 is = ",cur_b2,"A."

#A 125/250 V,3-wire distributor has an out-of-balance current of 50 A and larger load of 500 A.The balancer
#set has a loss of 375 W in each machine.Calculate the current in each of the balancer machines and output
#of main generator.
############################################################################################################

#Given
V_out = 250.0    #V
#Currents
cur_p = 500.0  #A
cur_o = 50.0   #A
cur_n = cur_p - cur_o  #A
#larger Load
load_p = cur_p*(V_out/2)/1000   #kW
#smaller Load
load_n = cur_n*(V_out/2)/1000   #kW
#Balancer loss
loss_b = 2*375.0/1000     #kW
#total load on generator
load_tot = load_p + load_n + loss_b
#current from main generator -> VI = W
cur_tot = load_tot*1000/V_out   #A

#Current in balancer sets
cur_b1 = cur_p - cur_tot     #A
cur_b2 = cur_o - cur_b1      #A
print "Current in balancer set 1 is = ",cur_b1,"A."
print "Current in balancer set 2 is = ",cur_b2,"A."
print "Output of main generator is = ",load_tot,"kW."

#The load on d.c 3-wire system with 500 V between outers consists of lighting current of 1500 A on the
#positive side and 1300 A on the negative side while motors connected across the outers absorb 500kW.
#Assuming that at this loading,the balancer machines have each a loss of 5kW,calculate the load on the
#main generator and on each of the balancer machines.
##########################################################################################################

#Given
cur_p = 1500.0   #A
cur_n = 1300.0   #A
V_out = 500.0    #V
load_ext = 500.0   #kW
loss_b = 2*5.0     #kW

#current through external load
cur_ext = load_ext*1000/V_out   #A
#larger load
load_p = cur_p*(V_out/2)/1000   #kW
#smaller load
load_n = cur_n*(V_out/2)/1000   #kW
#total load on generator
load_tot = load_p + load_n + loss_b + load_ext   #kW
#current from generator -> VI = W
cur_tot = load_tot*1000/V_out     #A
#current through neutral(out of balance)
cur_o = cur_p-cur_n      #A
#current through balancer sets
cur_b1 = (cur_p+cur_ext)-cur_tot   #A
cur_b2 = cur_o-cur_b1              #A
#load of balancer sets
load_b1 = cur_b1*(V_out/2)/1000    #kW
load_b2 = cur_b2*(V_out/2)/1000    #kW

print "Total load on main generator is = ",load_tot,"kW."
print "Load on Balancer set 1 is = ",load_b1,"kW."
print "Load on balancer set 2 is = ",load_b2,"kW."

#A d.c 3-wire system with 480 V across outers supplies 1200 A on the positive and 1000 A on the negative side.
#The balancer machines have each an armature resistances of 0.1W and take 10 A on no-load.Find
#(a)the voltage across each balancer and
#(b)the total load on the main generator and the current loading of each balancer machine.
#The balancer field windings are in series across the outers
################################################################################################################

#Given
V_out = 480.0   #V
#currents
cur_p = 1200.0  #A
cur_n = 1000.0  #A
cur_o = cur_p - cur_n  #A (out of balance)
#armature resistance 
res_arm = 0.1    #ohm
#no-load current
cur_nold = 10.0   #A

#Let us assume current Im flows through mtoring machine,then (200-Im) flows through generating machine.
#Let Vg and Vm be potential difference of 2 machines.

#Total losses in sets = no-load losses + Cu losses in two machines
#loss_set = V_out*cur_nold  + 0.1*Im^2+ 0.1*(200-Im)^2
#Vm*Im = Vg*Ig + loss_set
#Now, Vm = Eb+Im*Ra           Vg = Eb-Ig*Ra
Eb = V_out/2-res_arm*cur_nold

#Therefore, Vm = 239 + Im*0.1              and Vg = 239 - (200-Im)*0.1
#Hence,Equation is 
#(239+0.1*Im)*Im = [239 - (200-Im)*0.1]*(200-Im) + loss_set
#Simplified ->  239Im = 239*(200-Im)+4800

#Solving this equation
from sympy import Eq, var, solve
var('Im')    
eq = Eq(Eb*(2*Im-cur_o),V_out*cur_nold)
Im =  solve(eq)
Im = int(Im[0])
Ig = cur_o-Im
#Voltage across balancers

Vm = Eb + Im*res_arm     #V
Vg = Eb - Ig*res_arm     #V 

#Load on main generator
cur_load = cur_p - Ig    #A
print "Voltage across Balancer 1 is = ",round(Vg,2),"A."
print "Voltage across Balancer 2 is = ",round(Vm,2),"A."
print "Load current on main generator is = ",round(cur_load,2),"A."

#A d.c 3-wire system with 460V between outers supplies 250kW on the positive and 400kW on the negative side,
#the voltages being balanced.Calculate the voltage on the positive and negative side,the voltages being balanced.
#Calculate the voltage on the positive and negative sides repectively,if the neutral wire becomes disconnected
#from balancer set.
#################################################################################################################

#Given
V_mid = 230.0   #V
V_out = 460.0   #V
#loads
load_p = 250.0  #kW
load_n = 400.0  #kW
#resistance on positive side -> (V^2/R) = W
res_p = (V_mid*V_mid)/(load_p*1000)    #ohm

#resistance on negative side -> (V^2/R) = W
res_n = (V_mid*V_mid)/(load_n*1000)    #ohm

#Voltages after disconnecting balancer set
vol_p = (res_p/(res_p+res_n))*V_out      #V
vol_n = V_out - vol_p        #V

print "Voltage on positive side is = ",round(vol_p),"V."
print "Voltage on negative side is = ",round(vol_n),"V."

#A 2-wire system has the voltage at the supply end maintained at 500.The line is 3 km long.If the full-load
#current is 120 A,what must be the booster voltage and output in order that the far end voltage may also be 500 V.
#Take the resistance of the cable at the working temperature as 0.5ohm/kilometre.
####################################################################################################################

#Total resistance of line
res_tot = 0.5*3     #ohm
#Full load current
cur_full = 120.0    #A

#drop in the line-> V=IR
drop = res_tot*cur_full   #V

#Output of booster ->VI = W
output = drop*cur_full/1000   #kW

print "Terminal potential difference of the booster is = ",drop,"V."
print "Output of booster is = ",round(output,2),"kW."



