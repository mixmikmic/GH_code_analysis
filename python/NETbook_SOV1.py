import numpy as np
import scipy.io.idl
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import os

h = 6.626e-34 #Planck's constant
k = 1.381e-23 #Boltzmann's constant
Tcmb = 2.725 #CMB temperature

if not os.path.exists('OUTPUT_FILES'):
    os.makedirs('OUTPUT_FILES')

class BPStruct:

    def __init__(self, nu, filt, dnu, n):

        self.nu = nu
        self.filt = filt
        self.dnu = dnu
        self.n = n

def BP(nu, deltanu, sigma = 1.0, threshold = 0.001):

    xmax = 5.0 * deltanu + nu #Define minimum and maximum frequencies around the center
    xmin = nu - 5.0 * deltanu
    nx = xmax - xmin + 1
    if xmin < 0: 
        xmin = 0 
    x = np.arange(100*nx)/100. + xmin


    #set tophat to 1 where x is close to nu, 0 otherwise
    tophat = np.where((x >= nu-deltanu/2.0) * (x <= nu+deltanu/2.0), 
                      np.ones(len(x)), np.zeros(len(x)))

    xg = x - (xmax - xmin)/2.0
    kernel = np.exp(-xg**2/2./sigma**2)
    smallkernel = [] #get rid of small values in kernel that won't matter

    for i in range(len(kernel)):
        if kernel[i] > threshold: 
            smallkernel.append(kernel[i])

    result = np.correlate(tophat, smallkernel, mode = 'same')
    useband = []

    for i in range(len(result)):
        if result[i] > threshold: 
            useband.append(i)

    result = result[useband] #gets rid of values too small to matter
    nuvec = x[useband]
    result = result/np.max(result)
    dnu = nuvec[1] - nuvec[0]
    #frequency range:
    nulow = nuvec[0] #lower bound
    nuhigh = nuvec[len(nuvec)-1] #upper bound
    FreqRange = open("FreqRange.txt", "w")
    FreqRange.write(str(nulow) + "," + str(nuhigh))
    FreqRange.close()
    n = len(nuvec)

    return BPStruct(nu = nuvec, filt = result, dnu = dnu, n = n)

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

class OpticsElement:

    index = None
    trans = None
    etaa = None
    T = None
    name = None
    pload = None
    etato = None  

    def __init__(self, index, etaa, trans, T, name, pload = 0.0, etato =0.0):
        self.index = index
        self.trans = trans
        self.etaa = etaa
        self.T = T
        self.name = name
        self.pload = pload
        self.etato = etato

class SensStruct:

    def __init__(self, nu, deltanu, etadet, pwv, Tbath, psafety, nTES,
                 yieldFrac, opticsinfo, ptot, popt, patm, psky, Psat, G, Tc,
                 NEPG, NEPshot, NEPdicke, NEPphoton, NEPsky, NEPtot, NEPtotcmb,
                 arrayNEP, etasky, cmbfactor):

        vars = locals()
        for key in vars:
            if key != "self": self.__dict__[key] = vars[key]

def get_Tatm(savefile, pwv, interp_freqs): #interp_freqs=filter.nu; see function pload()
    savedata = scipy.io.idl.readsav(savefile)
    Pardo_pwv = savedata['jeffatm'].pwv[0]
    Pardo_freqs = savedata['jeffatm'].f[0]
    Pardo_Tatm = savedata['jeffatm'].tatm[0].transpose()
    first_matching_index = None

    for i in range(Pardo_pwv.size):
        #0.01 is threshold
        if np.abs(Pardo_pwv[i] - pwv) < 0.01:
            first_matching_index = i #index of first value in Pardo_pwv within 0.01mm of the set value for pwv. 
            break

    if first_matching_index == None:
        print('Choose from 0.25, 0.50, ... 3.50')
        return None

    if np.all(np.diff(Pardo_freqs) > 0):
        Tatm = np.interp(interp_freqs, Pardo_freqs, Pardo_Tatm[first_matching_index])
        return Tatm
    
    else:
        raise ValueError("Pardo frequencies are not strictly increasing")

def Pow(name, eta = 0.2, nu = None, deltanu = None, temp = 10.0, n0 = None,
            quiet = False, filt = None):
    
    if filt == None: 
        tophat = True

    else: tophat = False

    if tophat:
        if nu == None: 
            nu = 145.0
        if deltanu == None:
            if nu <= 100: 
                deltanu = 21.0
            if nu > 100 and nu <= 200: 
                deltanu = 34.0
            if nu > 200: 
                deltanu = 39.0

        bp = BP(nu, deltanu, sigma=sigma, threshold=0.001) 
        #bp defines frequency range to report out
        dnu = 1.0

    else:

        nu = filt.nu
        bp = filt.filt
        dnu = filt.dnu

                     
    nu = nu*1e9 #convert to Hz
    dnu = dnu*1e9 #convert to Hz
    
    x = h*nu/k/temp #See discussion above function Calculate()
    n0 = 1./(np.exp(x) - 1)
    
    
    power = eta*h*nu*n0*dnu*1e12*bp 
    #summing Planck blackbody power as a function of frequency in a narrow frequency band 
    pow = np.sum(power)
    meanetan0 = np.sum(eta*n0*bp)/np.sum(bp)
    meannu = np.sum(nu*bp)/np.sum(bp)
    deltanu = np.sum(bp*dnu)/1e9
    nu = meannu/1e9
    
    if not quiet:
        print(str(name) + ": eta*n0 = " +  str(meanetan0) + "; pow = " + str(pow) + " pW")
    return pow

f=open('Default_Input_SOV1.txt',"r")
InputFileName = f.name
lines=f.readlines()
InputValues=[]
for i in range(0, len(lines)-1):
    InputValues.append(lines[i].split(',')[1])
f.close()

def Calculate(nu, deltanu, quiet, etadet, pwv, etafeed, etamir, Tbath, etalensa, etawinr, 
              etawina, psafety, Twin, Tmir, T4K, Tstop, etafilta, sigma, nTES, yieldFrac, 
              etafiltr, etairfilt, totpow, TLP40K, TLP4K, TLP1K, T1K, T100mK, etalensr, n, flink, filt, savefile):
    
    if filt == None:
        if nu == None: 
            nu = 145.0
        if deltanu == None:
            if sigma==None: 
                sigma = 1.0
            if nu <= 100: 
                deltanu = 21.0
            if nu > 100 and nu <= 200: 
                deltanu = 34.0
            if nu > 200: 
                deltanu = 39.0
                    
        bp = BP(nu, deltanu, sigma=sigma, threshold=0.001)  
         #Above, we just use a simple tophat (bandpass) as a filter. 
        #In the future, will try more realistic filter functions.
    
    else:
        bp = filt
        nu = np.sum(bp.nu*bp.filt)/np.sum(bp.filt)
        deltanu = np.sum(bp.filt*bp.dnu)
        opticsinfo = []

    if n == None: #default n
        n = 2.7
        
    if flink == None: #default flink
        flink = 1.0
        
    #NOW GO THROUGH COMPONENTS IN OPTICAL PATH, INITIALIZING AS NEEDED 
    #Note that in this step, pload is not yet defined.

    opticsinfo = []
    
    #CMB
    opticsinfo.append(OpticsElement(index = 0, etaa = 1.0, trans = 1.0, T = Tcmb, name = 'CMB'))
    
    Tatm = get_Tatm(savefile, pwv, bp.nu)
    mean_delta_nu = np.sum(bp.filt*bp.dnu) #for a tophat, this is just deltanu
    mTatm = np.sum(Tatm*bp.filt*bp.dnu)/mean_delta_nu #average Tatm
    #print('bp.filt is: ' + str(bp.filt[1]))
    Tphys = np.max(Tatm) + 1.0
    if Tphys <= 250: 
        Tphys = 273.0
    etaatm = 1.0 - mTatm/Tphys

    #Atmosphere
    opticsinfo.append(OpticsElement(1,  etaa = 0.968, trans = 0.968, T = 8.736, name = 'ATM'))

    #First mirror
    opticsinfo.append(OpticsElement(2,  etaa = 0.990, trans = 0.990, T = 273.0, name = 'MIR1'))

    #Second mirror
    opticsinfo.append(OpticsElement(3,  etaa = 0.990, trans = 0.990, T = 273.0, name = 'MIR2'))
    
    #Third mirror
    #opticsinfo.append(OpticsElement(4,  etaa = 0.990, trans = 0.990, T = 273.0, name = 'MIR3'))

    #HDPE window 
    opticsinfo.append(OpticsElement(5,  etaa = 0.985, trans = 0.975, T = 265.0, name = "WIN"))

    #IR shaders
    opticsinfo.append(OpticsElement(6,  etaa = 0.999, trans = 0.999, T = 200.0, name = 'IR1'))

    #Low-pass filter 1
    opticsinfo.append(OpticsElement(7,  etaa = 0.990, trans = 0.940, T = 60.00, name = 'LP1'))

    #IR shaders
    opticsinfo.append(OpticsElement(8,  etaa = 0.999, trans = 0.999, T = 50.00, name = 'IR2'))

    #Low-pass filter 2
    opticsinfo.append(OpticsElement(9,  etaa = 0.990, trans = 0.940, T = 10.00, name = 'LP2'))

    #Lens 1
    opticsinfo.append(OpticsElement(10, etaa = 0.970, trans = 0.960, T = 4.500, name = 'lens1'))
    
    #Lens 2
    opticsinfo.append(OpticsElement(11, etaa = 0.970, trans = 0.960, T = 1.200, name = 'lens2'))

    #1K stop
    opticsinfo.append(OpticsElement(12, etaa = 0.650, trans = 0.650, T = 1.200, name = 'STOP'))

    #Lens 3
    opticsinfo.append(OpticsElement(13, etaa = 0.970, trans = 0.960, T = 1.200, name = 'lens3'))

    #Low-pass filter 3
    opticsinfo.append(OpticsElement(14, etaa = 0.990, trans = 0.940, T = 1.200, name = 'LP3'))

    #Low-pass filter 4
    opticsinfo.append(OpticsElement(15, etaa = 0.990, trans = 0.940, T = 1.200, name = 'LP4'))
    
    #Low-pass filter 5
    opticsinfo.append(OpticsElement(16, etaa = 0.990, trans = 0.940, T = 0.100, name = 'LP5'))
    
    #Detector
    opticsinfo.append(OpticsElement(17, etaa = 0.700, trans = 0.700, T = 0.100, name = 'DETS'))
    
    
    noptics = len(opticsinfo) #number of optical elements
    etasky = 1.0 #Starting value is 100% transmission

    for i in range(noptics): #step through each optical element to calculate total transmission.
        etasky = etasky * opticsinfo[i].trans

        
    #Now we calculate the loading power. 
    
    opticsinfo[0].pload = Pow(eta = etasky, filt = bp, temp = opticsinfo[0].T, 
        name = opticsinfo[0].name, quiet = True) #call function Pow() to calculate loading power of CMB

    ptot = opticsinfo[0].pload  #ptot is the loading power; Initially set to just that of the CMB

    for i in range(0, noptics-1): #-1 to avoid dividing by zero when calculating NEP_shot later. (pload for detectors at n=18 is zero) 
        pload(opticsinfo, i, bp, savefile)
        ptot = ptot + opticsinfo[i].pload #Step through each optical element and add up all the loading power terms to get total loading.

    psky = opticsinfo[0].pload + opticsinfo[1].pload #just loading power from CMB and atmosphere

    patm = opticsinfo[1].pload #just atmospheric loading

    popt = ptot - psky #loading of all optical elements from just the telescope (atmosphere and CMB excluded)

    #Now, we calculate the optimal values we want to know      
    Psat = psafety*ptot*1e-12
    #G = 2.3*Psat/Tbath #specifically, for flink=1.0 and n=2.7
    #Tc = Tbath*5.0/3.0 #specifically, for flink=1.0 and n=2.7
    Tc = Tbath*(n + 1)**(1./n)
    kappa = Psat/(Tc**n - Tbath**n)
    G=kappa*n*Tc**(n-1)
    
    #Noise Equivalent Power is calculated separately for each contributing source of noise. 
    NEPG = np.sqrt(4.0*k*(Tc**2)*G)# Assuming worst case scenario, f_link = 1.0; In the future, 
    #I will change this so that NEP will change as we vary f_link as a paramter. 
    tau = 0.5; #for calculating W/rt(Hz) use tau = 0.5s
    nu = nu*1e9 #put frequencies in Hz
    deltanu = deltanu*1e9
    NEPshot = np.sqrt(h*nu/tau*ptot*1e-12)
    NEPdicke = np.sqrt((ptot*1e-12)**2/deltanu/tau)
    NEPphoton = np.sqrt(NEPshot**2 + NEPdicke**2)
    NEPsky = np.sqrt(h*nu/tau*psky*1e-12 + (psky*1e-12)**2/deltanu/tau)

    NEPtot = np.sqrt(NEPdicke**2 + NEPshot**2 + NEPG**2)

    x = h*nu/k/Tcmb #consider flucns around Tcmb

    r = k*x**2*np.exp(x)/(np.exp(x) - 1.0)**2 * deltanu

    if totpow != 0: r = 2.*r 

    #refered back to sky but with units K and pW

    factorKpW = 1e-12/r/etasky 

    #refer back to sky and switch rt(Hz) to rt(s)

    r = 1.0/r/np.sqrt(2.0)/etasky 
    NEPtotcmb = NEPtot*r*1e6 #change units to microK (rt(s))
    ntot = yieldFrac*nTES 
    arrayNEP = NEPtotcmb/np.sqrt(ntot)
    nu = nu/1e9
    deltanu = deltanu/1e9
    
    #All output files are time stamped and the name of the input file used is written into the file
    
    OutputNET = open("NETValues.txt", "w")
    OutputNET.write("NET")
    OutputNET.write(" \n")
    OutputNET.write(str(arrayNEP))
    OutputNET.write(" \n")
    OutputNET.write(datetime.datetime.now().ctime())
    OutputNET.write(" \n")
    OutputNET.write("Input file: " + str(InputFileName))
    OutputNET.close()
    
      
    if not quiet: #"quiet" just determines whether all these values will be printed and the output files will be rewritten.
        #this file is rewritten each time Calculate() is called and then used later in NETbook to print relevant values.
        Output = open('OutputValues.txt', "w")
        Output.write("Psat, G, Tc, eta_system, NET") #etasystem is the total system efficiency from the sky down to the detectors
        Output.write(" \n")
        Output.write(str(round(psafety*ptot,3)) + "," + str(round(G*1e12,3)) + "," + str(round(Tc*1e3,3)) + "," +  str(round(etasky,3)) + "," + str(round(arrayNEP,3)))
        Output.write(" \n")
        Output.write(datetime.datetime.now().ctime())
        Output.write(" \n")
        Output.write("Input file: " + str(InputFileName))
        Output.close()
        
        #the same file as above, but the filename contains a time stamp.
        #This way, a record of output is automatically generated and not rewritten each time NETbook is run. 
        filename = 'OutputValues.' + str(datetime.datetime.now()) + ".txt"
        save_path = 'OUTPUT_FILES'
        Output1 = open(os.path.join(save_path, filename), "w")
        Output1.write("Psat, G, Tc, eta_system, NET") #etasystem is the total system efficiency from the sky down to the detectors
        Output1.write(" \n")
        Output1.write(str(round(psafety*ptot,3)) + "," + str(round(G*1e12,3)) + "," + str(round(Tc*1e3,3)) + "," +  str(round(etasky,3)) + "," + str(round(arrayNEP,3)))
        Output1.write(" \n")
        Output1.write(datetime.datetime.now().ctime())
        Output1.write(" \n")
        Output1.write("Input file: " + str(InputFileName))
        Output1.close()
    
        #Output2 is rewritten each time
        Output2 = open("OpticValues.txt", "w")
        Output2.write("Elementname, Emissivity, Efficiency, Temperature, Loading Power")
        Output2.write(" \n")
    
        for i in range(noptics - 1):
            Output2.write(str(opticsinfo[i].name) + "," + str(round((1 - opticsinfo[i].etaa),3)) + "," + str(round(opticsinfo[i].trans,3)) + "," + str(round(opticsinfo[i].T,3)) +  "," + str(round(opticsinfo[i].pload,3)))
            Output2.write(" \n")
    
        
        Output2.write(datetime.datetime.now().ctime())
        Output2.write(" \n")
        Output2.write("Input file: " + str(InputFileName))
        Output2.close() 

        #same as Output 2, but now we timestamp the filename so that a record is generated (not rewritten every time)
        save_path = 'OUTPUT_FILES'
        filename_optics = 'OpticValues.' + str(datetime.datetime.now()) + ".txt"
        Output3 = open(os.path.join(save_path, filename_optics), "w")
        Output3.write("Elementname, Emissivity, Efficiency, Temperature, Loading Power")
        Output3.write(" \n")
    
        for i in range(noptics - 1):
            Output3.write(str(opticsinfo[i].name) + "," + str(round((1 - opticsinfo[i].etaa),3)) + "," + str(round(opticsinfo[i].trans,3)) + "," + str(round(opticsinfo[i].T,3)) +  "," + str(round(opticsinfo[i].pload,3)))
            Output3.write(" \n")
    
        
        Output3.write(datetime.datetime.now().ctime())
        Output3.write(" \n")
        Output3.write("Input file: " + str(InputFileName))
        Output3.close() 
        
        print(" ") 
        print("Summary of Key Values:")
        print("")
        print("Psat = " + str(round(Psat*1e12,3)) + ' pW')
        print("G = " + str(round(G*1e12,3)) + ' pW/K')
        print("Tc = " + str(round(Tc*1e3,3))+' mK')  
        print("efficiency of detectors, etadet: " + str(etadet))
        print("overall efficiency from CMB sky down to detectors: " + str(round(etasky,3)))
        

        print("ATMOSPHERE PWV: " + str(pwv) + " mm")
        print("ATMOSPHERE mean Tatm: " + str(round(mTatm,3)) + " K")
        print(" ")
        print("The total of all the power terms is: " + str(round(ptot,3)) + " pW")
        print("Mirrors + window + lenses + stop are: " + str(round(popt,3)) + " pW")
        print("Psat safety factor is " + str(round(psafety,3)))
        print("Tbath = " + str(round(Tbath*1e3)) + " mK")
        print("Ndets = " + str(nTES) + " (Number of TESes; two TESes per horn)")
        print("yield = " + str(yieldFrac*100.0) + "%" + "(Assume " + str(round((1-yieldFrac)*100,1)) + "% of detectors are broken.)" )
        print("nu = " + str(nu) + "+/-" + str(deltanu) + " GHz")
        print("mean delta nu = " + str(mean_delta_nu) + " GHz")
        
        print("frequency range: ")
        g=open('FreqRange.txt',"r")
        line=g.readlines()
        print("Lowest frequency: " + str(line[0].split(',')[0]) + " GHz")
        print("Highest frequency: " + str(line[0].split(',')[1]) + " GHz")
        g.close()

        print("")
        print("The Total NEP is: " + str(round(NEPtot*1e17,3)) + 'e-17 W/rt(Hz)')
        print("The NEP from G is: " + str(round(NEPG*1e17,3)) + 'e-17 W/rt(Hz)')
        print("The shot noise NEP is: " + str(round(NEPshot*1e17,3)) + 'e-17 W/rt(Hz)')
        print("The dicke NEP is: " + str(round(NEPdicke*1e17,3)) + 'e-17 W/rt(Hz)')
        print("The Total Photon NEP is: " + str(round(NEPphoton*1e17,3)) + 'e-17 W/rt(Hz)')
        print("The total sky NEP is: " + str(round(NEPsky*1e17,3)) + 'e-17 W/rt(Hz)')
        print("CMB factor, r, (including eta and rt(2) ) is:  " +  str(round(r*1.0e-11,3)) + "e+17 muK rt(s)/W_det rt(Hz)")
        print("To convert pW at detector to K_CMB fluctuations from sky, " + "use : " + str(round(factorKpW,3)))
        print("")
        print("NET cmb is: " + str(round(NEPtotcmb)) + " microK rt (s)")
        print("Net array sensitivity: " + str(round(arrayNEP,3)) + " microK rt (s)")
        print("")
  
    return SensStruct(
        nu, deltanu, etadet, pwv, Tbath, psafety, nTES, yieldFrac, opticsinfo,
        ptot, popt, patm, psky, Psat, G, Tc, NEPG, NEPshot, NEPdicke, 
        NEPphoton, NEPsky, NEPtot, NEPtotcmb, arrayNEP, etasky, r)
    

def pload(opticsinfo, index, filt, savefile = 'atm_act.sav'):
    
    noptics = len(opticsinfo) #number of optical elements
    etahere = 1.0 #start with 100% transmission

    for i in range(index+1, noptics):  #steps through the transmission of each optical element to calculate total transmission up to (and excluding) the element we are considering.
        etahere = etahere * opticsinfo[i].trans   

    opticsinfo[index].etato = etahere * (1.0 - opticsinfo[index].etaa) #Etaa = 1-emissivity

    if opticsinfo[index].etato < 1e-7: 
        opticsinfo[index].etato = 1e-7 #set a realistic minimum

    if opticsinfo[index].name == 'ATM':#procedure for calculating the loading power for the Atmosphere
        #Tatm = get_Tatm(savefile, opticsinfo[index].T, filt.nu)
        Tatm = np.array([8.736 for i in range(len(filt.filt))])
        mTatm = np.sum(Tatm*filt.filt*filt.dnu)/np.sum(filt.filt*filt.dnu)
        Tphys = np.max(Tatm) + 1.0
        if Tphys < 250: 
            Tphys = 273.0 #set realistc minimum physical temperature; Increased from 250K; no longer winter season
        etatemporary = 1.0 - Tatm/Tphys #Atmospheric absorption factor in array form; See discussion regarding the function get_Tatm(). Note that this is a vector.
        etatoatm = 0*etatemporary #Initialize as an array of zeroes. 
        etasky = etahere #total transmission

        for i in range(len(etatemporary)):
            if etatemporary[i] > 0.001: #minimum threshold
                etatoatm[i] = etasky * (1.0 - etatemporary[i])

        for i in range(len(etatoatm)): #Need to make sure etatoatm is <1. 
            if etatoatm[i] > 1:
                etatoatm[i] = 0.9999

        opticsinfo[index].pload = Pow(
            eta = etatoatm, filt = filt, temp = Tphys, n0 = None, 
            name = 'ATM', quiet = True)  #call Pow() to calculate loading power
        opticsinfo[index].T = mTatm

    else: #for all other optical elements
        opticsinfo[index].pload = Pow(
            eta = opticsinfo[index].etato, filt = filt, 
            temp = opticsinfo[index].T, name = opticsinfo[index].name,
            quiet = True)

Set_filter = None
nu = float(InputValues[0])
deltanu = float(InputValues[1])
sigma = float(InputValues[17])

Calculate(nu = 147.0, 
          deltanu = 38.2, 
          quiet = False, 
          etadet = float(InputValues[3]), 
          pwv = float(InputValues[4]),
          etafeed = float(InputValues[5]), 
          etamir = float(InputValues[6]), 
          Tbath = 0.100, 
          etalensa = float(InputValues[8]), 
          etawinr = float(InputValues[9]), 
          etawina = float(InputValues[10]), 
          psafety = 3.0, 
          Twin = float(InputValues[12]), 
          Tmir = float(InputValues[13]), 
          T4K = float(InputValues[14]), 
          Tstop = float(InputValues[15]), 
          etafilta = float(InputValues[16]), 
          sigma = sigma,
          nTES = 43750,
          yieldFrac = 0.7, 
          etafiltr = float(InputValues[20]), 
          etairfilt = float(InputValues[21]), 
          totpow = float(InputValues[22]), 
          TLP40K = float(InputValues[23]), 
          TLP4K = float(InputValues[24]), 
          TLP1K = float(InputValues[25]), 
          T1K = float(InputValues[26]), 
          T100mK = float(InputValues[27]), 
          etalensr = 0.1, 
          n = 2.7, 
          flink = 1.0, 
          filt = Set_filter, 
          savefile = 'atm_act.sav')

Calculate(nu = 147.0, 
          deltanu = 30.0, 
          quiet = True, 
          etadet = float(InputValues[3]), 
          pwv = float(InputValues[4]),
          etafeed = float(InputValues[5]), 
          etamir = float(InputValues[6]), 
          Tbath = 0.100, 
          etalensa = float(InputValues[8]), 
          etawinr = float(InputValues[9]), 
          etawina = float(InputValues[10]), 
          psafety = 3.0, 
          Twin = float(InputValues[12]), 
          Tmir = float(InputValues[13]), 
          T4K = float(InputValues[14]), 
          Tstop = float(InputValues[15]), 
          etafilta = float(InputValues[16]), 
          sigma = sigma, 
          yieldFrac = 0.7, 
          etafiltr = float(InputValues[20]), 
          etairfilt = float(InputValues[21]), 
          totpow = float(InputValues[22]), 
          TLP40K = float(InputValues[23]), 
          TLP4K = float(InputValues[24]), 
          TLP1K = float(InputValues[25]), 
          T1K = float(InputValues[26]), 
          T100mK = float(InputValues[27]), 
          etalensr = float(InputValues[28]), 
          n = 2.7, 
          flink = 1.0, 
          filter = Set_filter, 
          savefile = 'atm_act.sav')
#Here we again use the values imported from the file "Default_Input.txt, 
#but any desired default could be set.

g=open('NETValues.txt',"r")
line=g.readlines()
NETdefault = float(line[1])
NETdefault_factor = 1/(NETdefault**2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
g.close() 

Set_filter = None
nu = float(InputValues[0])
deltanu = float(InputValues[1])
sigma = float(InputValues[17])

etafeed1 = []
for i in np.arange(0.55, 1.0, 0.01):
    etafeed1.append(i)  
  
NETarray = []     
           
for each in etafeed1: 
    etafeed = each
    Calculate(nu = nu, deltanu = deltanu, quiet = True, etadet = float(InputValues[3]), pwv = float(InputValues[4]),
        etafeed = etafeed, etamir = float(InputValues[6]), Tbath = float(InputValues[7]), 
        etalensa = float(InputValues[8]), etawinr = float(InputValues[9]), etawina = float(InputValues[10]), 
        psafety = float(InputValues[11]), Twin = float(InputValues[12]), Tmir = float(InputValues[13]), 
        T4K = float(InputValues[14]), Tstop = float(InputValues[15]), etafilta = float(InputValues[16]), 
        sigma = sigma, nTES = float(InputValues[18]), yieldFrac = float(InputValues[19]), 
        etafiltr = float(InputValues[20]), etairfilt = float(InputValues[21]), totpow = float(InputValues[22]), 
        TLP40K = float(InputValues[23]), TLP4K = float(InputValues[24]), TLP1K = float(InputValues[25]), 
        T1K = float(InputValues[26]), T100mK = float(InputValues[27]), etalensr = float(InputValues[28]), 
        n = None, flink = None, filter = Set_filter, savefile = 'atm_act.sav')
    
    g=open('NETValues.txt',"r")
    line=g.readlines()
    NEThere = float(line[1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    g.close() 
    NETarray.append(NEThere)
    
MappingRatio = []
for each in NETarray:
    MappingRatio.append(NETdefault_factor*each**2)

plt.plot(etafeed1, MappingRatio, 'k-')

coefficients = np.polyfit(etafeed1, MappingRatio, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(etafeed1)
print("Line of best fit: ")
print(polynomial)

plt.xlabel('etafeed')
plt.ylabel('Mapping Speed Ratio (NET/NETdefault)^2')

plt.title('Sensitivity to Stop Spillover')

plt.show()   

Set_filter = None
nu = float(InputValues[0])
deltanu = float(InputValues[1])
sigma = float(InputValues[17])

psafety1 = []
for i in np.arange(2., 6., 0.1):
    psafety1.append(i)
        
NETarray = []     
           
for each in psafety1:
    psafety = each
                  
    Calculate(nu = nu, deltanu = deltanu, quiet = True, etadet = float(InputValues[3]), pwv = float(InputValues[4]),
        etafeed = float(InputValues[5]), etamir = float(InputValues[6]), Tbath = float(InputValues[7]), 
        etalensa = float(InputValues[8]), etawinr = float(InputValues[9]), etawina = float(InputValues[10]), 
        psafety = psafety, Twin = float(InputValues[12]), Tmir = float(InputValues[13]), 
        T4K = float(InputValues[14]), Tstop = float(InputValues[15]), etafilta = float(InputValues[16]), 
        sigma = sigma, nTES = float(InputValues[18]), yieldFrac = float(InputValues[19]), 
        etafiltr = float(InputValues[20]), etairfilt = float(InputValues[21]), totpow = float(InputValues[22]), 
        TLP40K = float(InputValues[23]), TLP4K = float(InputValues[24]), TLP1K = float(InputValues[25]), 
        T1K = float(InputValues[26]), T100mK = float(InputValues[27]), etalensr = float(InputValues[28]), 
        n = None, flink = None, filter = Set_filter, savefile = 'atm_act.sav')
    
    g=open('NETValues.txt',"r")
    line=g.readlines()
    NEThere = float(line[1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    g.close() 
    NETarray.append(NEThere)
    
MappingRatio = []
for each in NETarray:
    MappingRatio.append(NETdefault_factor*each**2)

plt.plot(psafety1, MappingRatio, 'k-')

coefficients = np.polyfit(psafety1, MappingRatio, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(psafety1)
print("Line of best fit: ")
print(polynomial)

plt.xlabel('Safety factor, S')
plt.ylabel('Mapping speed ratio (NET/NET_default)^2')
plt.title('Sensitivity to Safety Factor')

plt.show()       

h=open('OutputValues.txt',"r")
line2=h.readlines()
print("AdvAct Forecast with: ")
print("Psat = " + str(line2[1].split(',')[0]) + " pW")     
print("G = " + str(line2[1].split(',')[1]) + " pW/K")     
print("Tc = " +  str(line2[1].split(',')[2]) + " mK")   
print("Total system efficiency = " + str(line2[1].split(',')[3]))     
print("NET = " + str(line2[1].split(',')[4]) + " microK rt (s)")

g=open('OpticValues.txt',"r")
line=g.readlines()

table = ListTable()
table.append(['Optical Element', 'Emissivity', 'Efficiency', 'Temperature (K) *', 'Loading Power (pW)'])
for i in range(1, len(line)-2):
    table.append([line[i].split(',')[0], line[i].split(',')[1], line[i].split(',')[2], line[i].split(',')[3], line[i].split(',')[4]])
g.close()
table

