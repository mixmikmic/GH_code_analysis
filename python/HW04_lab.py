######################################################################
## CDF for exponential distribution
## with parameter lambda -- from HW02_lab.ipynb
######################################################################
def exp_cdf(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([-1])

    return (1 - np.exp( -lam*arr ) )*(arr>0)

######################################################################
## probability density function for exponential distribution
## with parameter lambda -- from Lab04.ipynb
######################################################################
def exp_pdf(arr,lam):

    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for arr')
            return np.array([-1])
        
    return lam*np.exp(-lam*arr)*(arr>0)



