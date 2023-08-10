#Ref to the paper Calibrating and completing the volatility cube in the SABR model by G. Dimitroff and
#J. de Kock
import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def SABR(T,K,S0,sigma,alpha,beta,rho):  #T = T_m, time to maturity, All inputs are scalars
    
    S0K = S0*K
    lS0K = np.log(S0/K)
    
    z = (sigma/alpha)*((S0K)**((1-beta)/2))*(lS0K)
    x = np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
    
    denom = 1+(((1-beta)*lS0K)**2)/24 + (((1-beta)*lS0K)**4)/1920
    
    numer = 1 + T*((((1-beta)*alpha)**2)/(24*(S0K**(1-beta))) +      (rho*beta*sigma*alpha)/(4*(S0K**((1-beta)/2))) + ((sigma**2)*(2-3*(rho**2)))/24)
    
    imp_vol = (alpha*numer*(z/x))/(denom*(S0K**((1-beta)/2)))
    
    return (alpha*numer)/(denom*(S0K**((1-beta)/2))) if np.any(S0==K) else imp_vol

import pandas as pd
import xlrd

#file_input = xlrd.open_workbook('Swaptions Market Data.xlsx')
#Market_data = file_input.sheet_by_name('Swaptions Prof Tee')

m_data=pd.read_excel("Swaptions Market Data.xlsx",  sheetname= "Market Volatilities" )
m_data
#xls_file = pd.ExcelFile('Swaptions Market Data.xlsx')
#mkt_data = xls_file.parse('Swaptions Prof Tee')

#The first column is parsed as unicode. Change it to float for ease of calulations
for ind in range(0,18):
    m_data[-200][ind] = np.float64(m_data[-200][ind])

import scipy.stats as ss
#black76 formula to calculate price call price, T is time to maturity T_m
#This formula returns the undiscounted call price

def d1(T,K,F,sigma):
    return (np.log(F/K) + (sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(T,K,F,sigma):
    return (np.log(F/K) - (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def Black76(T,K,F,sigma):
    return (F * ss.norm.cdf(d1(T,K,F,sigma)) - K * ss.norm.cdf(d2(T,K,F,sigma)))

#USDLibor = xls_file.parse('USDLibor')
USDLibor= pd.read_excel("Swaptions Market Data.xlsx",  sheetname= "USDLibor" )
USDLibor['Mid'] = 0.5*(USDLibor['Ask']+USDLibor['Bid']) + 2.5 #added to prevent negative forward rates
USDLibor['Disc_Factor(0,yr)'] = 1/(1+USDLibor['Year']*USDLibor['Mid']/100)
USDLibor

D = []
D.append(1.0)
for ind in range(0,20):
    D.append(USDLibor['Disc_Factor(0,yr)'][ind])

#Linear interpolation of discount factors, where T1 - 1 < T < T1
def Interpolate(T,T1): 
    return float(T1-T)*D[T1-1] + float(T-(T1-1))*D[T1]

Zero_Disc_1M = np.zeros((4,11)) #To create a matrix of discount factors
for columns in range(0,11):
    for rows in range(0,4):
        Zero_Disc_1M[rows,columns] = Interpolate(columns+(1.0+(3.0*rows))/12,columns+1)
    
pd.DataFrame(Zero_Disc_1M)  
#The rows give the 1m,4m,7m,10m discount factors of each year to today
#for example the 1st row, last column gives us D(0,10yr1m)

def PVBasisPoint(year,DiscountFac):
    denom = 0.0
    for n in range(year+1):
        for m in range(0,4):
            denom =denom + DiscountFac[m][n]
    denom =denom - DiscountFac[0][0] + DiscountFac[0][year+1]
    return denom
def ForwardRateCalc(year,DiscountFac):
    forward = (DiscountFac[0][0] - DiscountFac[0][year+1])/(0.25*PVBasisPoint(year,DiscountFac))
    return forward

LiborForw = []
LiborForw.append(ForwardRateCalc(0,Zero_Disc_1M))  #The 1m1y forward rate
LiborForw.append(ForwardRateCalc(1,Zero_Disc_1M))  #The 1m2y forward rate
LiborForw.append(ForwardRateCalc(2,Zero_Disc_1M))
LiborForw.append(ForwardRateCalc(4,Zero_Disc_1M))
LiborForw.append(ForwardRateCalc(9,Zero_Disc_1M))

LiborForw

#For the 3m discount factors

Q_Zero_Disc_3M = np.zeros((4,11)) #To create a matrix of discount factors
for columns in range(0,11):
    for rows in range(0,4):
        Q_Zero_Disc_3M[rows,columns] = Interpolate(columns+(3.0+(3.0*rows))/12,columns+1)
    
pd.DataFrame(Q_Zero_Disc_3M)  
#The rows give the 3m,6m,9m,12m discount factors of each year to today
#for example the 1st row, last column gives us D(0,10yr3m)

LiborForw.append(ForwardRateCalc(0,Q_Zero_Disc_3M))    #The 3m1y forward rate
LiborForw.append(ForwardRateCalc(1,Q_Zero_Disc_3M))
LiborForw.append(ForwardRateCalc(2,Q_Zero_Disc_3M))
LiborForw.append(ForwardRateCalc(4,Q_Zero_Disc_3M))
LiborForw.append(ForwardRateCalc(9,Q_Zero_Disc_3M))

LiborForw  #rates are attached to the initial list

Q_Zero_Disc_6M = np.zeros((4,11)) #To create a matrix of discount factors

for columns in range(0,11):
    for rows in range(0,3):
        Q_Zero_Disc_6M[rows,columns] =  Q_Zero_Disc_3M[rows+1,columns]
for columns in range(0,10):
    Q_Zero_Disc_6M[3,columns] =  Q_Zero_Disc_3M[0,columns+1]

pd.DataFrame(Q_Zero_Disc_6M)  
#The rows give the 6m,9m,12m,1y3m discount factors of each year to today
#for example the 1st row, last column gives us D(0,10yr6m)

LiborForw.append(ForwardRateCalc(0,Q_Zero_Disc_6M))   #The 6m1y forward rate
LiborForw.append(ForwardRateCalc(1,Q_Zero_Disc_6M))
LiborForw.append(ForwardRateCalc(2,Q_Zero_Disc_6M))
LiborForw.append(ForwardRateCalc(4,Q_Zero_Disc_6M))
LiborForw.append(ForwardRateCalc(9,Q_Zero_Disc_6M))

LiborForw

Q_Zero_Disc_9M = np.zeros((4,11)) #To create a matrix of discount factors

for columns in range(0,11):
    for rows in range(0,3):
        Q_Zero_Disc_9M[rows,columns] =  Q_Zero_Disc_6M[rows+1,columns]
for columns in range(0,10):
    Q_Zero_Disc_9M[3,columns] =  Q_Zero_Disc_6M[0,columns+1]

pd.DataFrame(Q_Zero_Disc_9M)  
#The rows give the 9m,12m,1y3m,1y6m discount factors of each year to today
#for example the 1st row, last column gives us D(0,10yr9m)

LiborForw.append(ForwardRateCalc(1,Q_Zero_Disc_9M))   #The 9m2y forward rate
LiborForw.append(ForwardRateCalc(4,Q_Zero_Disc_9M))
LiborForw.append(ForwardRateCalc(9,Q_Zero_Disc_9M))

LiborForw

# change all values in the data to a common base(%).
# mkt_data.ix[:,1]    to return 1st column
# mkt_data[mkt_data.columns[0]] to return 1st column
# mkt_data.iloc[[2]] to return 3rd row

DataTable = m_data.copy()
for col in DataTable.columns: 
        DataTable[col] = m_data['ATM']/100 + m_data[col]/10000
DataTable['ATM'] = m_data['ATM']/100
DataTable

DataTable=DataTable.rename(columns = {'ATM': 0})
#Create strike grid
Str = np.zeros((18,11))
for i in range(18):
    for j in range(11):
        Str[i][j] = LiborForw[i] + 0.0001*(np.float64(DataTable.columns[j]))

df = pd.DataFrame(Str)
df     

#create market grid of volatilities
mkt_vol = np.zeros((18,11))
for i in range(18):
    for j in range(11):
        mkt_vol[i][j] = DataTable[DataTable.columns[j]][i]
mkt_vol[0][6]   

def obj_fun(abrs):      #This function solves for one tenor, maturity pair.
    beta = 0.5
    for j in range(11):  #for each m,n we have 11 sets of strikes
        S0K = LiborForw[0]*Str[0][j]
        lS0K = np.log(LiborForw[0]/Str[0][j])
    
        z = (abrs[0]/abrs[1])*((S0K)**((1-beta)/2))*(lS0K)
        x = np.log((np.sqrt(1-2*abrs[2]*z+z**2)+z-abrs[2])/(1-abrs[2]))
    
        denom = 1+(((1-beta)*lS0K)**2)/24 + (((1-beta)*lS0K)**4)/1920
    
        numer = 1 + (1.0/12)*((((1-beta)*abrs[1])**2)/(24*(S0K**(1-beta))) +          (abrs[2]*beta*abrs[0]*abrs[1])/(4*(S0K**((1-beta)/2))) +         ((abrs[0]**2)*(2-3*(abrs[2]**2)))/24)
    
        imp_vol = (abrs[1]*numer*(z/x))/(denom*(S0K**((1-beta)/2)))
        
        diff = imp_vol - mkt_vol[0][j]
        sum_sq_diff=0
        sum_sq_diff = sum_sq_diff+diff**2
    obj = math.sqrt(sum_sq_diff)
    
    return obj

#set starting guess for sigma,alpha,beta,rho
starting_guess = np.array([0.001,0.001,0])

from scipy.optimize import minimize
bnds = ( (0.001,None), (0.001,None), (-0.999,0.999) )

res = minimize(obj_fun, starting_guess, bounds = bnds, method='SLSQP')
res

res.x   #The values for sigma, alpha, beta, rho for the 1m1y pair.

#Let's do this for each row. Input time to maturity, Tm
TimePer=[]
for i in range(0,5):
    TimePer.append(1.0/12)
for i in range(0,5):
    TimePer.append(3.0/12)
for i in range(0,5):
    TimePer.append(6.0/12)
for i in range(0,3):
    TimePer.append(9.0/12)

def obj_fun_array(abrs,T,K,S0,mrkt,beta): 
    for j in range(11):  #for each m,n we have 11 sets of strikes, 11 rows, 11 pair of (m,n)
        
        S0K = S0*K[j]
        lS0K = np.log(S0/K[j])
    
        z = (abrs[0]/abrs[1])*((S0K)**((1-beta)/2))*(lS0K)
        x = np.log((np.sqrt(1-2*abrs[2]*z+z**2)+z-abrs[2])/(1-abrs[2]))
    
        denom = 1+(((1-beta)*lS0K)**2)/24 + (((1-beta)*lS0K)**4)/1920
    
        numer = 1 + T*((((1-beta)*abrs[1])**2)/(24*(S0K**(1-beta))) +          (abrs[2]*beta*abrs[0]*abrs[1])/(4*(S0K**((1-beta)/2))) +         ((abrs[0]**2)*(2-3*(abrs[2]**2)))/24)
    
        imp_vol = (abrs[1]*numer*(z/x))/(denom*(S0K**((1-beta)/2)))
        
        diff = imp_vol - mrkt[j]
        sum_sq_diff=0
        sum_sq_diff = sum_sq_diff+diff**2
    obj_ar = math.sqrt(sum_sq_diff)
    
    return obj_ar

sigma = []
alpha = []
rho = []

def Calibrate(guess,T,K,S0,mrkt,beta):
    for i in range(18):
        x0 = guess
        bnds = ( (0.001,None), (0.001,None), (-0.999,0.999) )
    
        result = minimize(obj_fun_array, x0, (T[i],K[i],S0[i],mrkt[i],beta), 
                       bounds = bnds, method='SLSQP')
        sigma.append(result.x[0])
        alpha.append(result.x[1])
        rho.append(result.x[2])
    

Calibrate(starting_guess,TimePer,Str,LiborForw,mkt_vol,0.5)

print sigma

Parameters = np.zeros((18,3))
for i in range(18):
    Parameters[i][0] = sigma[i]
    Parameters[i][1] = alpha[i]
    Parameters[i][2] = rho[i]
    
df1=pd.DataFrame(Parameters)

df1.columns = ['Sigma', 'Alpha', 'Rho']
df1.index = m_data.index
df1

Strike_Range = np.linspace(0, 0.09, 400, endpoint=True)
ImpVol_1m1y = SABR(TimePer[0],Strike_Range,LiborForw[0],sigma[0],alpha[0],0.5,rho[0])
ImpVol_9m10y = SABR(TimePer[17],Strike_Range,LiborForw[17],sigma[17],alpha[17],0.5,rho[17])

plt.plot(Strike_Range,ImpVol_9m10y)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('Volatility smile for 9M10Y contract')
plt.show()

PVBP = []
PVBP.append(PVBasisPoint(0,Zero_Disc_1M))  #The 1m1y discount factors
PVBP.append(PVBasisPoint(1,Zero_Disc_1M))  #The 1m2y discount factors
PVBP.append(PVBasisPoint(2,Zero_Disc_1M))
PVBP.append(PVBasisPoint(4,Zero_Disc_1M))
PVBP.append(PVBasisPoint(9,Zero_Disc_1M))
PVBP.append(PVBasisPoint(0,Q_Zero_Disc_3M))  #The 3m1y discount factors
PVBP.append(PVBasisPoint(1,Q_Zero_Disc_3M))  #The 3m2y discount factors
PVBP.append(PVBasisPoint(2,Q_Zero_Disc_3M))
PVBP.append(PVBasisPoint(4,Q_Zero_Disc_3M))
PVBP.append(PVBasisPoint(9,Q_Zero_Disc_3M))
PVBP.append(PVBasisPoint(0,Q_Zero_Disc_6M))  #The 6m1y discount factors
PVBP.append(PVBasisPoint(1,Q_Zero_Disc_6M))  #The 6m2y discount factors
PVBP.append(PVBasisPoint(2,Q_Zero_Disc_6M))
PVBP.append(PVBasisPoint(4,Q_Zero_Disc_6M))
PVBP.append(PVBasisPoint(9,Q_Zero_Disc_6M))
PVBP.append(PVBasisPoint(1,Q_Zero_Disc_9M))  #The 9m2y discount factors
PVBP.append(PVBasisPoint(4,Q_Zero_Disc_9M))
PVBP.append(PVBasisPoint(9,Q_Zero_Disc_9M))

def SABR_VOL(T,K,S0,sig,alph,bet,ro):
    Imp_Vol = np.zeros((18,11))
    for i in range(18):
        for j in range(11):
            Imp_Vol[i][j] = SABR(T[i],K[i][j],S0[i],sig[i],alph[i],bet,ro[i])
    return Imp_Vol

def Black76_Price(T,K,F,Vol,BP):
    B76price = np.zeros((18,11))
    for i in range(18):
        for j in range(11):
            B76price[i][j] = BP[i]*Black76(T[i],K[i][j],F[i],Vol[i][j])
    return B76price

Implied_Volatilities = SABR_VOL(TimePer,Str,LiborForw,sigma,alpha,0.5,rho)

pd.DataFrame(Implied_Volatilities)

Arbitrage_free_price = Black76_Price(TimePer,Str,LiborForw,Implied_Volatilities,PVBP)

pd.DataFrame(Arbitrage_free_price)







def DisplcedDiffusion(T,K,F,Vol,BP,b):  #input b as a decimal
    DDprice = np.zeros((18,11))
    coeff = (1-b)/b
    for i in range(18):
        for j in range(11):
            DDprice[i][j] = BP[i]*Black76(T[i],K[i][j]+coeff*F[i],F[i]/b,Vol[i][j]*b)
    return DDprice

from scipy.optimize import root
#DDVol be the volatility to solve for
def root_fun_array(Param,T,K,F,BP,Price):
    coeff = (1-Param[1])/Param[1]
    fun_val = []
    fun_val.append(BP*Black76(T,K+coeff*F,F/Param[1],Param[0]*Param[1])- Price)
    fun_val.append(BP*Black76(T,K+coeff*F,F/Param[1],Param[0]*Param[1])- Price)
    return fun_val

DD_Vol = np.zeros((18, 11))
colms = range(11)
colms.pop(5)  #Exclude ATM price
DD_b=np.zeros((18,11))

start_value = np.array([0.001,0.5])

def Solve_Root(guess,T,K,F,BP,Price):
    for i in range(18):
        for j in colms:
            result = root(root_fun_array, guess, (T[i],K[i][j],F[i],BP[i],Price[i][j]), method = 'hybr')
            DD_Vol[i][j] = (result.x[0])
            DD_b[i][j] = (result.x[1])
    

Solve_Root(start_value,TimePer,Str,LiborForw,PVBP,Arbitrage_free_price) 

dfDD_Vol=pd.DataFrame(DD_Vol)
print dfDD_Vol

dfDD_b=pd.DataFrame(DD_b)
print dfDD_b

root_fun_array(start_value,TimePer[0],Str[0][0],LiborForw[0],PVBP[0],Arbitrage_free_price[0][0])

