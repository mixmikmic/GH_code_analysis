def cosine(x, amp, f, p, o):
    '''
    Utility function to fit nonlinearly
    '''
    return amp*np.cos(2*np.pi*f*(x-p))+o

def cosine_sat(x,*params):
    o = params[-1]
    p = params[-2]
    f = params[-3]
    amps = params[:len(params)-3]
    
    to_return = zeros_like(x)
    for i, amp in enumerate(amps):
        #each amp corresponds to the ith harmonic
        to_return += cosine(x,amp,f*(i+1),p,0)
    
    return to_return+o

def calc_mod_sat(data, num_harms = 1, nphases = 24,periods = 2):
    '''
    Need to change this so that it:
    - first tries to fit only the amplitude and phase
        - if that doesn't work, estimate amp and only fit phase
    - then do full fit
    '''

    #pull internal number of phases
    #nphases = self.nphases

    #only deal with finite data
    #NOTE: could use masked wave here.
    finite_args = np.isfinite(data)
    data_fixed = data[finite_args]
    
    popt = None
    
    if len(data_fixed) > 4:
        #we can't fit data with less than 4 points
        #make x-wave
        x = np.arange(nphases,dtype=data_fixed.dtype)[finite_args]

        #make guesses
        #amp of sine wave is sqrt(2) the standard deviation
        g_a = np.sqrt(2)*(data_fixed.std())
        #offset is mean
        g_o = data_fixed.mean()
        #frequency is such that `nphases` covers `periods`
        g_f = periods/nphases
        #guess of phase is from first data point (maybe mean of all?)
        g_p = nan
        i = 0
        while not isfinite(g_p):
            g_p = x[i]-np.arccos((data_fixed[i]-g_o)/g_a)/(2*np.pi*g_f)
            i+=1
        
        #make guess sequence
        if num_harms == 1:
            amps = [g_a]
        if num_harms == 2:
            # https://www.wolframalpha.com/input/?i=expand+1-cos(x)%5E4
            amps = [-4/5*g_a, -1/5*g_a]
            g_p+=3*pi/2
        if num_harms == 3:
            # https://www.wolframalpha.com/input/?i=expand+1-cos(x)%5E6
            amps = [-15/22*g_a, -6/22*g_a,-1/22*g_a]
            g_p+=3*pi/2
        if num_harms == 4:
            amps = [-56/93*g_a, -28/93*g_a,-8/93*g_a,-1/93*g_a]
            g_p+=3*pi/2
            
        pguess = amps + [g_f,g_p,g_o]
        try:
            popt,pcov = curve_fit(cosine_sat,x,data_fixed,p0=array(pguess))
        except RuntimeError as e:
            #if fit fails, put nan
            print(e)
            mod = np.nan
            res = nan
        except TypeError as e:
            print(e)
            print(data_fixed)
            mod = np.nan
            res = nan
        else:
            if len(popt)==4:
                opt_a,opt_f,opt_p,opt_o = popt
                opt_a = np.abs(opt_a)
                #if any part of the fit is negative, mark as failure
                if opt_o - opt_a < 0:
                    mod = np.nan
                else:
                    #calc mod
                    mod = 2*opt_a/(opt_o+opt_a)
            else:
                mod = np.nan
            res = (data_fixed-cosine_sat(x,*popt))**2
            res = res.sum()
    else:
        mod = np.nan
        res = nan
    
    
    
    return popt, res, mod

def cosine(x, amp, f, p, o):
    '''
    Utility function to fit nonlinearly
    '''
    return amp*np.cos(2*np.pi*f*(x-p))+o

def cosine_sat(x,*params):
    o = params[-1]
    p = params[-2]
    f = params[-3]
    amps = params[:len(params)-3]
    
    to_return = zeros_like(x)
    for i, amp in enumerate(amps):
        #each amp corresponds to the ith harmonic
        to_return += cosine(x,amp,f*(i+1),p,0)
    
    return to_return+o

#Testing the function
x = linspace(0,23,1024)
#popt = array([ -9.23688107e+02,  -2.15499865e+01,   1/12,   12*pi,   2.31774918e+03])
popt = array([  -56/128, -28/128, -8/128, -1/128,   1/12,  0, 93/128])
fig, ax = subplots(1,1,figsize=(12,12))
fit = cosine_sat(x,*popt)
ax.plot(x,fit,'r-', label = 'Cosine Sat Func')
amps = popt[:len(popt)-3]
# plot each harmonic seperately
for i, amp in enumerate(amps):
    #each amp corresponds to the ith harmonic
    o = popt[-1]
    p = popt[-2]
    f = popt[-3]
    mod = 2*abs(amp)/(o+abs(amp))
    ax.plot(x,cosine(x,amp,f*(i+1),p,-o*(amp/abs(amps).sum())),'--',label='Mod = {:.2f}\nAmp = {:.2f}'.format(mod, amp))

fit_direct = 1 - (cosine(x, 1/2, f, p , 1/2))**4
ax.plot(x, fit_direct,'k.', label= '$\cos(x)^4$')
ax.legend()

def calc_mod_sat(data, num_harms = 1, nphases = 30,periods = 2):
    '''
    Need to change this so that it:
    - first tries to fit only the amplitude and phase
        - if that doesn't work, estimate amp and only fit phase
    - then do full fit
    '''

    #pull internal number of phases
    #nphases = self.nphases

    #only deal with finite data
    #NOTE: could use masked wave here.
    finite_args = np.isfinite(data)
    data_fixed = data[finite_args]
    
    popt = ones(4)*nan
    pguess = ones(4)*nan
    res = np.nan
    
    if len(data_fixed) > 3+num_harms:
        #we can't fit data if number of parameters exceeds number of points
        #make x-wave
        x = np.arange(nphases,dtype=data_fixed.dtype)[finite_args]
        
        #make guess sequence
        if num_harms == 1:
            #make guesses
            #amp of sine wave is sqrt(2) the standard deviation
            g_a = np.sqrt(2)*(data_fixed.std())
            #offset is mean
            g_o = data_fixed.mean()
            #frequency is such that `nphases` covers `periods`
            g_f = periods/nphases
            #guess of phase is from median, note that amp is made negative due to the saturation model.
            g_p = median((x-np.arccos(-(data_fixed-g_o)/g_a)/(2*np.pi*g_f))[:nphases//periods])
            amps = [1]
        if num_harms == 2:
            amps = [4/5, 1/5]
        if num_harms == 3:
            amps = [15/22, 6/22,1/22]
        if num_harms == 4:
            amps = [56/93, 28/93,8/93,1/93]
            
        if num_harms > 1:
            #first do the fit for num_harms 1
            popt, res, pguess = calc_mod_sat(data, num_harms = 1, nphases = nphases,periods = periods)
            g_a, g_f, g_p, g_o = popt
            #if the fitted amp is positive that just means we need to adjust the phase
            # and make the amp negative
            if g_a > 0:
                g_p += nphases/periods/2
                g_a = -g_a
            
        # for saturated data we expect the peaks to be negative, so shift phase and make amplitudes negative.
        pguess = concatenate((array(amps)*g_a, [g_f, g_p, g_o]))
        if isfinite(pguess.all()):
            try:
                popt,pcov = curve_fit(cosine_sat,x,data_fixed,p0=pguess)
            except RuntimeError as e:
                #if fit fails, put nan
                pass
            except TypeError as e:
                print(e)
                print(data_fixed)
            else:
                res = ((data_fixed-cosine_sat(x, *popt))**2).sum()
    
    
    
    return popt, res, pguess

def find_best_num_harms(y, max_harm = 4, verbose =False):
    '''
    A function to find the best number of harmonics to fit the data
    '''
    # set up the stopping criterion
    amps_valid = False
    mods_valid = False
    res_old = np.inf
    popt_old = np.nan
    # loop through having 2 to 4 harmonics, fitting each one
    for i in list(range(2, max_harm+1))+[1]:
        popt, res, pguess = calc_mod_sat(y, i)
        if verbose:
            print('pguess', pguess)
            print('popt', popt)
        # pull the offset
        o = popt[-1]
        # and the amps
        amps = popt[:len(popt)-3]
        # check to see if amps are positive
        amps_pos = (amps < 0).all()
        # check to see that the amplitudes are ordered such that the strength decreasing with increasing harmonics
        amps_valid = amps_pos and (amps.argsort() == arange(len(amps))).all()
        # check the modulation depths as well.
        mods = 2*abs(amps)/(o+abs(amps))
        #make sure modulation depths are between 1 and 0, inclusive
        mods_valid = np.logical_and(mods <= 1.0, mods >= 0.0).all()
        
        if res > res_old:
            # if the residuals have increased, don't update
            if verbose:
                print('Residuals have increased')
            # break
        elif mods_valid and (amps_valid or len(popt) == 4):
            # if the residuals have decreased and the parameter are valid, update
            if verbose:
                print('Residuals have decreased and popt valid, updating...')
            res_old = res
            popt_old = popt
        else:
            if verbose:
                print('Residuals have decreased but popt invalid')

        if verbose:
            print('{:.3e}'.format(res_old))

    if verbose:
        print('Final residuals = {:.3e}'.format(res_old))
    
    return popt_old, res_old, pguess



