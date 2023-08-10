from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
#
class Pulsar:
# initialize class
    def __init__(self, filename):
        hdulist = fits.open(filename)
        orb_elem = hdulist[1].data
        self.name = orb_elem.field("NAME")[0]
        self.ra = orb_elem.field("RA")[0]
        self.dec = orb_elem.field("DEC")[0]
        self.binary = orb_elem.field("BINARY")
        self.pbinary = orb_elem.field("PBINARY")[0]
        self.pbdot = orb_elem.field("PBDOT")[0]
        self.epoch_type = orb_elem.field("EPOCH_TYPE")[0]
        self.binaryepoch = orb_elem.field("BINARYEPOCH")[0]
        self.axsini = orb_elem.field("AXSINI")[0]
        self.periapse = orb_elem.field("PERIAPSE")[0]
        self.apsidalrate = orb_elem.field("APSIDALRATE")[0]
        self.eccentricity = orb_elem.field("ECCENTRICITY")[0]
        self.egress = orb_elem.field("EGRESS")[0]
        self.ingress = orb_elem.field("INGRESS")[0]
        hist = hdulist[2].data
        self.tstart = hist.field("TSTART")[hist.field("DETECTED") == 1]
        self.tstop = hist.field("TSTOP")[hist.field("DETECTED") == 1]
        self.psrtime = hist.field("PSRTIME")[hist.field("DETECTED") == 1]
        self.frequency = hist.field("FREQUENCY")[hist.field("DETECTED") == 1]
        self.frequency_err = hist.field("FREQUENCY_ERR")[hist.field("DETECTED") == 1]
        self.amplitude = hist.field("AMPLITUDE")[hist.field("DETECTED") == 1]
        self.amplitude_err = hist.field("AMPLITUDE_ERR")[hist.field("DETECTED") == 1]
        self.numharm = hist.field("NUMHARM")[hist.field("DETECTED") == 1]
        hdulist.close()
        # tmin and tmax set the plotting bounds in MJD
        self.tmin = np.ndarray.min(self.psrtime) - 20.
        self.tmax = np.ndarray.max(self.psrtime) + 20.
        self.fmin = (np.ndarray.min(self.frequency) - np.ndarray.max(self.frequency_err)) * 1000.0
        self.fmax = (np.ndarray.max(self.frequency) + np.ndarray.max(self.frequency_err)) * 1000.0
        self.amin = np.ndarray.min(self.amplitude) - np.ndarray.max(self.amplitude_err)
        self.amax = np.ndarray.max(self.amplitude) + np.ndarray.max(self.amplitude_err)
    def printinfo(self):
        if self.binary == 'Y' :
            print('Source name {} ra={} dec={}'.format(self.name, self.ra, self.dec))
            print('The following ephemeris is used to determine pulsar emission frequencies')
            print('Binary orbital period is {} days'.format(self.pbinary))
            if np.abs(self.pbdot) > 0:
                print('Binary orbital period derivative is {} days/day'.format(self.pbdot))
            if self.epoch_type =='T':
                print('Pi/2 Epoch is {} JED'.format(self.binaryepoch))
            else:
                print('Periastron Epoch is {} JED'.format(self.binaryepoch))
            print('AXSIN(i) is {} light-sec'.format(self.axsini) )
            print('Long. of periastron is {} deg'.format(self.periapse))
            print('Eccentricity is {}'.format(self.eccentricity))
            if self.egress > 0:
                print('Source is eclipsing:')
                print('Egress is {}'.format(self.egress))
                print('Ingress is {}'.format(self.ingress))
        else:
            print('Source name {} ra={} dec={}'.format(self.name, self.ra, self.dec))
            print('No ephemeris is used and frequencies are not corrected for the pulsar orbit')
    def setminmax(self,fmin=None,fmax=None,amin=None,amax=None):
        # Use user defined vertical plot ranges.  Automatically set using current tmin and tmax 
        # if vertical plot ranges are unspecified
        if fmin is None:
            self.fmin = (np.ndarray.min(self.frequency[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])-                 np.ndarray.max(self.frequency_err[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])) * 1000.0
        else:
            self.fmin = fmin
        if fmax is None:
            self.fmax = (np.ndarray.max(self.frequency[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])+                np.ndarray.max(self.frequency_err[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])) * 1000.0
        else:
            self.fmax = fmax
        if amin is None:
            self.amin = np.ndarray.min(self.amplitude[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])-                 np.ndarray.max(self.amplitude_err[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])
        else:
            self.amin = amin
        if amax is None:
            self.amax = np.ndarray.max(self.amplitude[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])+                np.ndarray.max(self.amplitude_err[(self.psrtime >= self.tmin) * (self.psrtime < self.tmax)])
        else:
            self.amax = amax
    def plot(self,tmin=None,tmax=None,fmin=None,fmax=None,amin=None,amax=None):
        #Use user supplied tmin tmax [MJD]
        if tmin is not None:
            self.tmin = tmin
        if tmax is not None:
            self.tmax = tmax
        self.setminmax(fmin = fmin, fmax = fmax, amin = amin, amax = amax)
        plt.figure(1)
        plt.subplot(211)        
        plt.ylabel('Frequency [mHz]')
        plt.axis([self.tmin,self.tmax,self.fmin,self.fmax])
        plt.errorbar(self.psrtime, self.frequency*1000.,self.frequency_err*1000.,fmt='.')
        plt.subplot(212)
        plt.axis([self.tmin,self.tmax,self.amin,self.amax])
        plt.xlabel('Time [MJD]')
        plt.ylabel('12-25 keV Pulsed Flux')
        plt.errorbar(self.psrtime, self.amplitude,self.amplitude_err,fmt='.')
        plt.show()

oao1657 = Pulsar('oao1657.fits')

oao1657.plot()

oao1657.printinfo()

np.ndarray.min(oao1657.psrtime)

np.ndarray.min(oao1657.psrtime)

oao1657.plot(amin = .2, fmin = 27.)

print(oao1657.fmin)

print(oao1657.tmin)

print(oao1657.fmin)

oao1657.tmin = 55500

oao1657.plot()



