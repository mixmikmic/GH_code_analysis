import numpy as np
import pandas as pd
import math

#Determining the implied probability of default in CDS from observed market spreads
def CDS_Implied_PD(R, T, dt, N, RR, Lam, Mkt_Spread):
    #Create empty data frame and create index by period
    df= pd.DataFrame()
    index = np.arange(0,T+dt,dt)
    df['period'] = index
    df = df.set_index(index)
    df['Notional'] = N
    df['disc_factor'] = np.exp(-R * df['period'])
    
    #Create column to hold the guess for the implied hazard rate
    df['lambda'] = Lam
    
    #Calculate probability of survival and probability of default based on hazard rate guess
    df['P_Survival'] = np.exp(df['period']*-df['lambda'])
    df['P_Default'] = df['P_Survival'].shift(1) - df['P_Survival']
    df.loc[0,'P_Default'] = 0

    df['premium_leg'] = df['Notional'] * df['disc_factor'] * Mkt_Spread * dt *df['P_Survival']
    df.loc[0,'premium_leg'] = 0
    df['default_leg'] = df['Notional'] * (1-RR) * df['P_Default'] * df['disc_factor']
    pv_premium_leg = df['premium_leg'].sum()
    pv_default_leg = df['default_leg'].sum()
    mtm = pv_default_leg - pv_premium_leg
    return mtm, Lam, df

def CDS_root_find(R, T, dt, N, RR, Mkt_Spread):    
    #Calculation of implied hazard rate
    Lam = 0.1 #Initial Estimate for Lambda
    count = 100000  #number of attempts to find value
    mtm, Lam, df = CDS_Implied_PD(R, T, dt, N, RR, Lam, Mkt_Spread)
    while abs(mtm) > (.0001*N) and count > 0:
        if mtm > (.0001 * N):
            Lam = Lam -.0001
        else:
            Lam = Lam + .0001
        mtm, Lam, df1 = CDS_Implied_PD(R, T, dt, N, RR, Lam, Mkt_Spread)        
        count -= 1        
    return mtm, Lam, count

#Run functions to determine implied probability of default
#input contract variables and CDS market spread
'''
R = Risk Free Rate
T = Time to maturity in years
dt = Payment Frequency in fraction of a year  .25 = quarterly, .5 = semiannually
N = Notional Amount
RR = Recovery Rate
Mkt_Spread = Observed CDS Spread in %
'''
R = .05
T = 5
dt = .25
N = 1000000
RR = .40
Mkt_Spread = .0133

mtm, Lam, count = CDS_root_find(R, T, dt, N, RR, Mkt_Spread)
implied_survival_probability = math.exp(-Lam * T)

print("The implied survival probability is %.4f" %implied_survival_probability)
print("The implied default probability is %.4f" %(1-implied_survival_probability))



