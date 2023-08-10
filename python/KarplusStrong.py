import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython
import scipy.interpolate as interp

sample_rate = 44100

def p(a):
    return IPython.display.Audio( a, rate=sample_rate, autoplay = True )

def kp_standalone():
    freq = 220
    # F = sample / ( N + .5 )
    # so N + .5 = sample / F

    burstlen = int( (sample_rate / freq + 0.5)* 2 )

    samplelen = sample_rate * 3
    result = np.zeros( samplelen )
    noise = np.random.rand( burstlen ) * 2 - 1

    result[0:burstlen] = noise
    delay = noise

    pos = burstlen
    
    filtAtten = 0.4;
    filtWeight = 0.5

    filtAtten = filtAtten / 100  / ( freq / 440 )

    while( pos < samplelen ):
        dpos = pos % burstlen
        dpnext = (pos + 1 ) % burstlen
        dpfill = (pos - 1) % burstlen

        # Simple averaging filter
        filtval = ( filtWeight * delay[ dpos ] + (1.0 - filtWeight ) * delay[ dpnext ] )  * (1.0 - filtAtten)

        result[ pos ] = filtval
        delay[ dpfill ] = filtval
    
        pos = pos + 1

    return result
        
p( kp_standalone() )
# filtAtten

class KSSynth:
    def __init__(self):
        self.pos = 0
        self.filtWeight = 0.5
        self.filtAtten = 3.0
        
        self.filterType = "weightedOneSample"
        
        self.setFreq( 220 )
        
        self.initPacket = "random"
        
    def setFreq( self, freq ):
        self.freq = freq
        self.burstlen = int( ( sample_rate / freq + 0.5 ) * 2 )
        self.filtAttenScaled = self.filtAtten / 100 / ( self.freq / 440 )
        
    def trigger( self, freq ):
        self.setFreq( freq )
        self.delay = []
        ls = np.linspace( 0, self.burstlen-1, self.burstlen ) / self.burstlen
        if( self.initPacket == "random" ):
            self.delay = np.random.rand( self.burstlen ) * 2 - 1   
        if( self.initPacket == "square" ):
            mp = int(self.burstlen/2)
            self.delay = np.zeros( self.burstlen )
            self.delay[ :mp ] = 1
            self.delay[ mp: ] = -1
        if( self.initPacket == "saw" ):
            self.delay = ls * 2 - 1
        if( self.initPacket == "noisysaw" ):
            self.delay = ls  * 1 - 0.5 + np.random.rand( self.burstlen ) - 0.5
        if( self.initPacket == "sin" ):
            self.delay = np.sin( ls  * 2 * np.pi )
        if( self.initPacket == "sinChirp" ):
            lse = np.exp( ls * 2 ) * 3
            self.delay = np.sin( lse * 2 * np.pi )
        if( len( self.delay ) == 0 ):
            print( "Didn't grok ", self.initPacket )
        
    def adjFrequency( self, freq ):
        """This is different than trigger in that it keeps current waves and interps them to a new freq"""
        oldbl = self.burstlen
        olddel = self.delay
        self.setFreq( freq )
        
        olddi = interp.interp1d( np.arange( 0, oldbl ), olddel )
        newy = np.arange( 0, self.burstlen ) * (oldbl-1) / (self.burstlen-1)
        self.delay = olddi( newy )
        
    def step( self ):
        dpos = self.pos % self.burstlen
        dpnext = ( self.pos + 1 ) % self.burstlen
        dpfill = ( self.pos - 1 ) % self.burstlen

        # Simple averaging filter
        fw = self.filtWeight;
        fa = self.filtAttenScaled;
        filtval = -1000;
        if( self.filterType == "weightedOneSample" ):
            filtval = ( fw * self.delay[ dpos ] + ( 1.0 - fw ) * self.delay[ dpnext ] )  * ( 1.0 - fa )
        if( filtval == -1000 ):
            filtval = 0
            print( "Filtval misset ", self.filterType )
            
        self.delay[ dpfill ] = filtval
    
        self.pos = self.pos + 1      
        return filtval

k = KSSynth()
k.trigger( 440 )
# print( np.average( k.delay ) )
print( np.sqrt( np.sum( [ i * i for i in k.delay ])) / k.burstlen )
[k.step() for i in range( 2000 )]
# print( np.average( k.delay ) )
print( np.sqrt( np.sum( [ i * i for i in k.delay ])) / k.burstlen )
[k.step() for i in range( 2000 )]
# print( np.average( k.delay ) )
print( np.sqrt( np.sum( [ i * i for i in k.delay ])) / k.burstlen )



#ds = [ k.step() for i in range(sample_rate)]
#print( k.filtAttenScaled, " ", k.burstlen )
p(ds)
#plt.plot( ds )

k = KSSynth()
packets = [ "random", "square", "saw", "noisysaw", "sin", "sinChirp" ]
npk = len( packets ) 
f = []
for i in range( 13 ):
    fm = pow( 2, i/12.0 ) 
    k.initPacket = packets[ i % npk ]
    k.trigger( fm * 220 )
    res = [ k.step() for i in range( int(sample_rate/2) ) ]
    f = f + res
    
p( f )

k = KSSynth();
f = []
k.filtAtten = 0.1
fr = 440;
k.trigger( 440 )
sr10 = int( sample_rate / 500 )
mul = 1.005
for i in range( 800 ):
    res = [ k.step() for i in range( sr10 )]
    fr = fr * mul
    k.adjFrequency( fr )
    f = f + res
    if( i == 250 ):
        #fr = 440
        mul = 0.999
p( f )

np.linspace( 0, 70, 71 )

plt.plot( f[ 0:500 ] )



