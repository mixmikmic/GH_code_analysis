#Module Imports
import numpy as np
import numpy.random as npr
import math
import scipy as sp
from scipy import stats

'''DEFINITION OF VARIABLES
    S0 - Stock Price at T=0
    E - Strike Price
    T - Time in Years
    R - Risk Free Rate
    SIGMA - Volatility
    DT - Time Step = T/N
    DF - Discount Factor = e^-RT
    I - Number of Simulations
    P - Discrete Sampling Frequency for Asian Options 
        252/Annual, 126/SemiAnnual, 63/Quarterly, 21/Monthly, 1/Continuous
'''

S0 = 100
E=100
T=1
R=0.05
SIGMA=0.20
I=10000
P= 21 #Discrete Sampling Frequency 252/Annual, 126/SemiAnnual, 63/Quarterly, 21/Monthly, 1/Continuous
N=252

'''OPTION VALUATION - W/ ANTITHETIC VARIANCE REDUCTION W/ MILSTEIN SCHEME - 
ASIAN OPTIONS - FIXED AND FLOATING STRIKE'''
def option_valuation(S0, E, T, N, SIGMA, R, I, P):    
    DT = T/N   #Time Step
    DF = math.exp(-R*T)  #Discount Factor    
#GENERATE RANDOM NUMBERS - ANTITHETIC VARIANCE REDUCTION
    PHI = npr.standard_normal((N,int(I/2))) 
    PHI = np.concatenate((PHI, -PHI), axis=1)     
#SET UP EMPTY ARRAYS AND SET INITIAL VALUES    
    S = np.zeros_like(PHI)  #Array to Capture Asset Value Path
    S[0] = S0
    P_AVG=np.zeros_like((S))  #Array to Capture Arithmetic Average Sample
    G_AVG=np.zeros_like((S))  #Array to Capture Geometric Average Sample
#CREATE FOR LOOP TO GENERATE SIMULATION PATHS - MILSTEIN METHOD
    for t in range (1,N):
        S[t]=S[t-1]*(1+R*DT+(SIGMA*PHI[t]*np.sqrt(DT))+(SIGMA**2)*(0.5*(((PHI[t]**2)-1)*DT)))
#Heaviside Function to Determine When to Take an Average
#On sample date the average is taken and stored in the appropriate array
        Mod = int(t) % P 
        if Mod == 0:
            P_AVG [t-1] = np.mean(S[(t-(P)):t], axis=0)
            G_AVG [t-1] = sp.stats.gmean(S[(t-P):t], axis=0)
            
        P_AVG[-1] = np.mean(S[(-P):N], axis=0)
        P_AVG_Payoff = np.sum(P_AVG[0:N], axis=0) / (N/P)
        
        G_AVG[-1] = sp.stats.gmean(S[(-P):N], axis=0)
        G_AVG_Payoff = np.sum(G_AVG[0:N], axis=0) / (N/P)  
 
#Calculation of Discounted Expected Payoff for Asian Options - Arithmetic Mean 
    Call_Value_Asian = DF * np.sum(np.maximum((P_AVG_Payoff) - E, 0)) / I
    print( "Value of Fixed Strike Asian Call Option - Arithmetic Average =  %.3f" %Call_Value_Asian)
    Put_Value_Asian = DF * np.sum(np.maximum(E - (P_AVG_Payoff), 0)) / I
    print( "Value of Fixed Strike Asian Put Option - Arithmetic Average = %.3f" %Put_Value_Asian) 

#Calculation of Discounted Expected Payoff for Asian Options - Geometric Mean
    Call_Value_Asian_GEO = DF * np.sum(np.maximum((G_AVG_Payoff) - E, 0)) / I
    print( "Value of Asian Fixed Strike Call Option - Geometric Average = %.3f" %Call_Value_Asian_GEO)
    Put_Value_Asian_GEO = DF * np.sum(np.maximum(E - (G_AVG_Payoff), 0)) / I
    print( "Value of Asian Fixed Strike Put Option - Geometric Average = %.3f" %Put_Value_Asian_GEO)

#Calculation of Discounted Expected Payoff for Asian Options - Geometric Mean - Floating Strike
    Call_Value_Asian_GEO_Float_Strike = DF * np.sum(np.maximum(S[-1] - (G_AVG_Payoff), 0)) / I
    print( "Value of Asian Floating Strike Call Option - Geometric Average = %.3f" %Call_Value_Asian_GEO_Float_Strike)
    Put_Value_Asian_GEO_Float_Strike = DF * np.sum(np.maximum((G_AVG_Payoff) - S[-1], 0)) / I
    print( "Value of Asian Floating Strike Put Option - Geometric Average = %.3f" %Put_Value_Asian_GEO_Float_Strike)

#Calculation of Discounted Expected Payoff for Asian Options - Arithmetic Mean - Floating Strike 
    Call_Value_Asian_Float_Strike = DF * np.sum(np.maximum(S[-1] - (P_AVG_Payoff), 0)) / I
    print( "Value of Asian Floating Strike Call Option - Arithmetic Average = %.3f" %Call_Value_Asian_Float_Strike)
    Put_Value_Asian_Float_Strike = DF * np.sum(np.maximum((P_AVG_Payoff) - S[-1], 0)) / I
    print( "Value of Asian Floating Strike Put Option - Arithmetic Average = %.3f" %Put_Value_Asian_Float_Strike) 

option_valuation(S0, E, T, N, SIGMA, R, I, P)



