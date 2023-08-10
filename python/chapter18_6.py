#Variable declaration
VCC=10;             #Supply voltage, V
RC=1.0;               #Collector resistor, kΩ
RB=47.0;              #Base resistor, kΩ
beta=100.0;           #Base current amplification factor
VBE=0.7;            #Base-emitter voltage, V

#Calculation
IC_sat=VCC/RC;              #Collector saturation current, mA
IB=IC_sat/beta;             #Base current, mA
V=IB*RB+VBE;                #Input voltage, V

#Result
print("Input voltage required to saturate the transistor switch=%.1fV."%V);

#Variable declaration
VCC=10;             #Supply voltage, V
RC=1.0;               #Collector resistor, kΩ
ICBO=10.0;            #Collector leakage current, μA
V_knee=0.7;         #Knee voltage, V


#Calculation
#(i)
IC=ICBO;                    #Collector current, μA
VCE=VCC-(ICBO/1000)*RC;            #Collector-emitter voltage, V

print("(i) The collector emitter voltage at cut-off=%.2fV."%VCE);

#(ii)
#Since, saturation current=IC_sat=(VCC-V_knee)/RC;         
VCE=V_knee;                    #Collector-emitter voltage, V

print("(ii) The collector emitter voltage at saturation=%.1fV."%VCE);

#Variable declaration
VCC=10;             #Supply voltage, V
RC=1;               #Collector resistor, kΩ
VBB=2;              #Supply voltage to base, V
RB=2.7;             #Base resistor, kΩ
V_knee=0.7;         #Knee voltage, V
VBE=0.7;            #Base-emitter voltage, V

#Calculation
#(i)
IB=round((VBB-VBE)/RB,2);           #Base current, mA
Ic_sat=(VCC-V_knee)/RC;             #Collector saturation current, mA
beta_min=Ic_sat/IB;                 #Minimum value of base current amplification factor
print("(i) Minimum β=%.1f."%beta_min);

#(ii)
VBB=1;                       #Supply voltage to base(changed), V
beta=50;                     #Base current amplification factor
IB=(VBB-VBE)/RB;             #Base current, mA
IC=beta*IB;                  #Collector current,mA

if(IC<Ic_sat):
    print("(ii) The transistor will not be saturated.");
else:
    print("(ii) The transistor will be saturated.");

#Variable declaration
R2=10;                  #Resistor R2, kΩ
R3=10;                  #Resistor R3, kΩ
C1=0.01;                #Capacitor of 1st transistor, μF
C2=0.01;                #Capacitor of 2nd transistor, μF

#Calculation
R=R2*1000;                   #Resistance, Ω
C=C1*10**-6;                 #Capacitance, F
T=round((1.4*R*C)*1000,2);   #Time period,m sec
f=1/(T*10**-3);              #Frequency, Hz
f=f/1000;                    #Frequency, kHz

#Result
print("Time period of the square wave=%.2f m sec."%T);
print("Time frequency of the square wave=%d kHz."%f);

#Variable declaration
R=10;               #Resistance in differentiating circuit, kΩ
C=2.2;              #Capacitance in differentiating circuit, μF
d_ei=10;            #Change in input voltage, V
dt=0.4;             #Time in which change occurs, s

#Calculation
eo=R*1000*C*10**-6*d_ei/dt


#Result
print("The output voltage=%.2fV."%eo);

#Variable declaration
Vin_peak=12;                #Peak value of input voltage, V
V_D=0.7;                    #Forward bias voltage of diode, V

#Calculation
Vout_peak=Vin_peak-V_D;         #Peak value of output voltage, V

#Result
print("The peak output voltage=%.1fV."%Vout_peak);

#Variable declaration
Vin_peak=10;                #Peak value of input voltage, V
R=1;                        #Input resistor, kΩ
RL=4;                       #Load resistor, kΩ

#Calculation
Vout_peak=(Vin_peak*RL)/(R+RL);         #Peak output voltage, V


#Result
print("The peak output voltage=%dV."%Vout_peak);

#Variable declaration
Vin=-10;                #Input voltage, V
V_D=0.7;                #Forward bias voltage of the diode, V
R=1;                    #Resistance, kΩ


print("The diode will be forward biased for the negative half-cycle of input signal.");
Vout=-V_D;              #Output voltage, V
V_R=Vin-(-V_D);         #Voltage across resistor R, V

#Result
print("The output voltage=%.1fV."%Vout);
print("The voltage across R=%.1fV."%V_R);

#Variable declaration
V_F=0.7;                        #Forward bias voltage of diode, V
R=200.0;                          #Input resistor of the circuit,  Ω
RL=1.0;                           #Load resistor, kΩ
Vin_peak=10.0;                    #Peak input voltage, V


#Calculations

#Positive half-cycle:
print("During the positive half cycle, the diode is foward biased and can be replaced by battery of %.1fV."%V_F);
print("Therefore, Vout=%.1fV."%V_F);

#Negative half-cycle:
print("During the negative half cycle, the diode is reverse biased and hence behaves as an open circuit.");
Vout_peak=RL*(-Vin_peak)/(R/1000+RL);
print("Therefore, Vout_peak=%.2fV."%Vout_peak);

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import sin
from math import pi

V_biasing=10.0;                        #Biasing voltage, V
vin=[30*sin(t/10.0) for t in range(0,(int)(2*pi*10))]      #input voltage  waveform, V

plt.subplot(211)
plt.plot(vin);
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');

vout=[];                             #Output voltage waveform, V
for v in vin[:]:
    if(v-V_biasing)>0 :              #Diode is forward biased.
        vout.append(v-V_biasing);
    else:                            #Diode is reverse biased.
        vout.append(0);

plt.subplot(212)        
plt.plot(vout);
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

Vin=[];      #Input voltage waveform, V
t1=50;         #Assumed time interval, s
t2=100;         #Assumed time interval, s
V_biasing=10;  #Biasing voltage, V
for t in range(0,151):                  #time interval from 0s to 151s
    if(t<=t1):                      
        Vin.append(15);               #Value of input voltage for time 0 to t1 seconds 
    elif(t<=t2 and t>t1):
        Vin.append(-30);             #Value of input voltage for time t1 to t2 seconds
    else :
        Vin.append(15);             #Value of input voltage after t2 seconds

plt.subplot(211)
plt.plot(Vin);
plt.xlim([0,160])
plt.ylim([-35,20])
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');


vout=[];                 #Output voltage waveform, V
for v in Vin[:]:                  #Loop iterating input voltage 
    if(v<=0):
        vout.append(0);               #Diode reverse biased
    else:
        vout.append(v-V_biasing);      #Diode forward biased

plt.subplot(212)
plt.plot(vout);
plt.xlim(0,160)
plt.ylim(-35,20)
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

Vin=[];      #Input voltage waveform, V
t1=50;         #Assumed time interval, s
t2=100;         #Assumed time interval, s
V_biasing=5;  #Biasing voltage, V
for t in range(0,151):                  #time interval from 0s to 151s
    if(t<=t1):                      
        Vin.append(10);               #Value of input voltage for time 0 to t1 seconds 
    elif(t<=t2 and t>t1):
        Vin.append(-10);             #Value of input voltage for time t1 to t2 seconds
    else :
        Vin.append(0);

plt.subplot(211)        
plt.plot(Vin);
plt.xlim(0,101)
plt.ylim(-20,20)
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');


vout=[];                 #Output voltage waveform, V
for v in Vin[:]:                  #Loop iterating input voltage 
    if(v<=0):
        vout.append(v);               #Diode reverse biased
    else:
        vout.append(v-V_biasing);      #Diode forward biased

plt.subplot(212)
plt.plot(vout);
plt.xlim(0,101)
plt.ylim(-20,20)
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

Vin=[];      #Input voltage waveform, V
t1=50;         #Assumed time interval, s
t2=100;         #Assumed time interval, s
V_D1=0.6;          #Forward Biasing voltage of the 1st diode, V
V_D2=0.6;          #Forward Biasing voltage of the 2nd diode, V
for t in range(0,151):                  #time interval from 0s to 151s
    if(t<=t1):                      
        Vin.append(10);               #Value of input voltage for time 0 to t1 seconds 
    elif(t<=t2 and t>t1):
        Vin.append(-10);             #Value of input voltage for time t1 to t2 seconds
    else :
        Vin.append(0);

plt.subplot(211);
plt.plot(Vin);
plt.xlim(0,110)
plt.ylim(-20,20)
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');

vout=[];                 #Output voltage waveform, V
for v in Vin[:]:                  #Loop iterating input voltage 
    if(v<=0):
        vout.append(-V_D1);               #Diode D1 forward biased, 
    else:
        vout.append(V_D2);      #Diode D2 forward biased

plt.subplot(212)        
plt.plot(vout);
plt.xlim(0,110)
plt.ylim(-1,1)
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import sin
from math import pi

VZ=20;                   #Assumed zener voltage, V
VF=0.7;          #Assumed forward biasing voltage of the zener diode, V
Vin=[];      #Input voltage waveform, V
for t in range(0,(int)(2*pi*10)):                  #time interval from 0s to 151s
    Vin.append(30*sin(t/10.0));

plt.subplot(211)
plt.plot(Vin);
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');


vout=[];                 #Output voltage waveform, V
for v in Vin[:]:                  #Loop iterating input voltage 
    if(v<=-VF):
        vout.append(-VF);               #Zener diode forward biased, 
    elif(v>=VZ):
        vout.append(VZ);        #Input voltage exceeds zener voltage
    else:
        vout.append(v);           #Zener diode reverse biased

plt.subplot(212)
plt.plot(vout);
plt.xlim([0,80])
plt.ylim([-1,40])
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import sin
from math import pi

VZ1=20;                   #Assumed zener voltage, V
VF1=0.7;          #Assumed forward biasing voltage of the zener diode, V
VZ2=20;                   #Assumed zener voltage, V
VF2=0.7;          #Assumed forward biasing voltage of the zener diode, V
Vin=[];      #Input voltage waveform, V
for t in range(0,(int)(2*pi*10)):                  #time interval from 0s to 151s
    Vin.append(30*sin(t/10.0));
    
plt.subplot(211)
plt.plot(Vin);
plt.xlabel('t-->');
plt.ylabel('Vin(V)');
plt.title('Input waveform');


vout=[];                 #Output voltage waveform, V
for v in Vin[:]:                  #Loop iterating input voltage 
    if(v<=-(VZ1+VF2)):
        vout.append(-(VZ1+VF2));               #Zener diode forward biased, 
    elif(v>=VZ2+VF1):
        vout.append(VZ2+VF1);        #Input voltage exceeds zener voltage
    else:
        vout.append(v);           #Zener diode reverse biased
plt.subplot(212)
plt.plot(vout); 
plt.xlim([0,80])
plt.ylim([-40,40])
plt.xlabel('t-->');
plt.ylabel('Vout(V)');
plt.title('Output waveform');



