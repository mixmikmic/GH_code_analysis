import sys
import numpy as np
import math
import ceo
import matplotlib.pyplot as plt
import IPython
get_ipython().magic('matplotlib inline')
import scipy.io as sio

## Generic simulation parameters
VISU = True   # show graphic displays

#Compute the rms over each segment
def segment_rms(WF, P):
    rms_seg = np.sqrt(np.sum((WF[np.newaxis,:,np.newaxis]*P)**2,axis=1)/P.sum(axis=1))
    return rms_seg

print '----->  Initializing CEO objects....'

# Telescope parameters
D = 25.
nPx = 14*26+1
# nPx = n*nLenslet + 1
radial_order = 4
n_zern = (radial_order+1)*(radial_order+2)/2
gmt = ceo.GMT_MX(D,nPx,M1_radial_order=radial_order,M2_radial_order=radial_order)

# Initialize on-axis star for performance evaluation
ongs = ceo.Source("I",zenith=0.,azimuth=0., rays_box_size=D,rays_box_sampling=nPx,rays_origin=[0.0,0.0,25])

# number of SH WFS Guide Stars and position
N_GS = 1
if N_GS > 1:
    alpha =6*60.  # radius of circle where GSs are located [in arcsec]
    zenith_angle  = np.ones(N_GS)*alpha*math.pi/180/3600 # in radians
    azimuth_angle = np.arange(N_GS)*360.0/N_GS # in degrees
else:
    alpha = 0.
    zenith_angle = 0.
    azimuth_angle = 0.
    
# GS for SH wavefront sensing
gs = ceo.Source("R",zenith=zenith_angle,azimuth=azimuth_angle*math.pi/180,
                    rays_box_size=D,rays_box_sampling=nPx,rays_origin=[0.0,0.0,25])

# SH WFS parameters
nLenslet = 26  # number of sub-apertures across the pupil
n = 14         # number of pixels per subaperture
detectorRes = 2*n*nLenslet/2
BINNING = 2
wfs = ceo.ShackHartmann(nLenslet, n, D/nLenslet,N_PX_IMAGE=2*n,BIN_IMAGE=BINNING,N_GS=N_GS)

#Initialize asterism for performance evaluation in the science field

#SF_N_GS = 16
#sf_alpha =[15.,30.]  # radius of science field (arcsec)
#SF_N_SEP = len(sf_alpha)
#sf_zenith_angle = np.tile(sf_alpha, SF_N_GS)*math.pi/180/3600 # in radians
#sf_azimuth_angle = np.repeat(np.arange(SF_N_GS)*360.0/SF_N_GS, SF_N_SEP) # in degrees
#sfxx = np.tile(sf_alpha, SF_N_GS) * np.cos(sf_azimuth_angle*math.pi/180)  #in arcsec
#sfyy = np.tile(sf_alpha, SF_N_GS) * np.sin(sf_azimuth_angle*math.pi/180)  #in arcsec

sfx = np.arange(-45.,55.,15.) # in arcsec
sfy = np.arange(-45.,55.,15.) # in arcsec
sfxx, sfyy = np.meshgrid(sfx, sfy)
sf_zenith_angle = np.sqrt(sfxx**2 + sfyy**2) * ceo.constants.ARCSEC2RAD
sf_azimuth_angle = np.arctan2(sfyy,sfxx)

sfgs = ceo.Source("I",zenith=sf_zenith_angle,azimuth=sf_azimuth_angle,#*math.pi/180,
                  rays_box_size=D,rays_box_sampling=nPx,rays_origin=[0.0,0.0,25])

if VISU == True:
    fig,ax1 = plt.subplots()
    fig.set_size_inches((4,4))
    ax1.plot(sfxx,sfyy, '*', markersize=10)
    ax1.set_xlim([-100,100])
    ax1.set_ylim([-100,100])
    ax1.grid()

#Initialize SPS guide stars and sensors
SPStype = 'DFS'  #Choose between "ideal" or "DFS"

N_GS_PS = 3
alpha_ps = 6.0*60.  # radius of circle where GSs are located [in arcsec]
zenith_angle_ps  = np.ones(N_GS_PS)*alpha_ps*math.pi/180/3600 # in radians
azimuth_angle_ps = np.arange(N_GS_PS)*360.0/N_GS_PS # in degrees
gsps = ceo.Source("J",zenith=zenith_angle_ps,azimuth=azimuth_angle_ps*math.pi/180,
                    rays_box_size=D,rays_box_sampling=nPx,rays_origin=[0.0,0.0,25])

if SPStype == 'ideal':
    # Idealized Segment Piston Sensors
    ps = ceo.IdealSegmentPistonSensor(gmt, gsps)
elif SPStype == 'DFS':
    # Dispersed Fringe Sensors
    ps = ceo.DispersedFringeSensor(gmt.M1,gsps)#,nyquist_factor=2)
    ps.lobe_detection = 'gaussfit' #peak_value'

gmt.reset()   # Telescope perfectly phased

# Calibrate AO SH WFS slope null vector
gs.reset()
gmt.propagate(gs)
wfs.calibrate(gs,0.8)

if VISU == True:
    plt.imshow(wfs.flux.host(shape=(nLenslet*N_GS,nLenslet)).T,interpolation='none')

print "pupil sampling: %d pixel"%nPx
print "SH Pixel scale: %.3farcsec"%(wfs.pixel_scale_arcsec)
sh_fov = wfs.pixel_scale_arcsec*wfs.N_PX_IMAGE/BINNING
print "SH Field of view: %.3farcsec"%(sh_fov)

# Calibrate DFS
gsps.reset()
if SPStype == 'DFS':
    ps.calibrate(gsps,gmt)
    ps.reset()
    gsps.reset()

# Calibrate SPS reference vector (corresponding to field-dependent aberrations)
gmt.propagate(gsps)
ph_fda = gsps.phase.host(units='micron').T
SPSmeas_ref = ps.piston(gsps, segment='edge')
#print np.array_str(SPSmeas_ref*1e9, precision=4, suppress_small=True)

if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    fig.suptitle('Field-dependent aberrations (um)', fontsize=20)
    imm = ax.imshow(ph_fda, interpolation='None')
    fig.colorbar(imm, orientation='horizontal', shrink=0.6)

### Show DFS refernce imagettes
def show_sps_imagettes():
    dataCube = ps.get_data_cube(data_type='fftlet')

    fig, ax = plt.subplots(ps.camera.N_SIDE_LENSLET,ps.camera.N_SIDE_LENSLET)
    fig.set_size_inches((12,12))
    xvec = np.arange(0,ps.camera.N_PX_IMAGE,10)
    for k in range(gsps.N_SRC*12):   
        (ax.ravel())[k].imshow(np.sqrt(dataCube[:,:,k]), cmap=plt.cm.gist_earth_r, origin='lower')
        (ax.ravel())[k].autoscale(False)
        if ps.INIT_ALL_ATTRIBUTES == True:
            (ax.ravel())[k].plot(xvec, xvec*ps.pl_m[k] + ps.pl_b[k], 'y')
            (ax.ravel())[k].plot(xvec, xvec*ps.pp_m[k] + ps.pp_b[k], 'y--')
            for pp in range(3):
                c1 = Circle((sps.blob_data[k,pp,1], sps.blob_data[k,pp,0]),radius=np.sqrt(2)*ps.blob_data[k,pp,2], 
                            color='b', fill=False)    
                (ax.ravel())[k].add_patch(c1)
        (ax.ravel())[k].set_title('%d'%(k%12), fontsize=12)

    for k in range(ps.camera.N_SIDE_LENSLET**2):
        (ax.ravel())[k].axis('off')
        
if VISU == True:
    show_sps_imagettes()

# Field aberrations in the on-axis performance directions (ongs)
ongs.reset()
gmt.propagate(ongs)
ph_fda_on = ongs.phase.host(units='micron')
Wref = np.rollaxis( ongs.wavefront.phase.host(units='nm', shape=(1,ongs.N_SRC,ongs.n*ongs.m)),1,3)
on_rms0 = ongs.wavefront.rms()*1e9
print '--> WF RMS on-axis: %3.2f nm wf RMS'%on_rms0

if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    fig.suptitle('Field-dependent aberrations (um wf)', fontsize=20)
    imm = ax.imshow(ph_fda_on, interpolation='None')
    fig.colorbar(imm, orientation='horizontal', shrink=0.6)

# Field aberrations in the off-axis performance directions (sfgs)
sfgs.reset()
gmt.propagate(sfgs)
sf_rms0 = sfgs.wavefront.rms()*1e9
print '--> max WF RMS at the edge of the field: %3.2f nm WF RMS'%np.max(sf_rms0)

if VISU == True:
    fig, ax2 = plt.subplots()
    fig.set_size_inches((5,4))
    contp = ax2.contourf(sfx, sfy, sf_rms0.reshape(len(sfx),-1))
    clb = fig.colorbar(contp, ax=ax2)
    ax2.grid()
    ax2.tick_params(labelsize=12)
    ax2.set_xlabel('field angle [arcsec]', fontsize=15)
    clb.set_label('nm WF RMS', fontsize=15)
    clb.ax.tick_params(labelsize=12)

#### Define $\rho$ and $\theta$ coordinates on each mirror segment mask (on-axis direction)
Zobj = ceo.ZernikeS(radial_order)
P = np.rollaxis( np.array( ongs.rays.piston_mask ),0,3)

## Find center coordinates (in pixels) of each segment mask
u = np.arange(ongs.n)
v = np.arange(ongs.m)
x,y = np.meshgrid(u,v)
x = x.reshape(1,-1,1)
y = y.reshape(1,-1,1)
xc = np.sum(x*P,axis=1)/P.sum(axis=1)
yc = np.sum(y*P,axis=1)/P.sum(axis=1)
#plt.plot(alpha_off/60,xc[(0,3,6),:].T,'-+')

## Preliminary estimation of radius (in pixels) of each segment mask (assuming that there is no central obscuration)
Rs = np.sqrt(P.sum(axis=1)/np.pi)

## Polar coordinates
rho   = np.hypot(   x - xc[:,np.newaxis,:], y - yc[:,np.newaxis,:])   #temporal rho vector
theta = np.arctan2( y - yc[:,np.newaxis,:], x - xc[:,np.newaxis,:]) * P

## Estimate central obscuration area of each segment mask
ObsArea = np.sum(rho < 0.9*Rs[:,np.newaxis,:] * ~P.astype('bool'), axis=1)

## Improve estimation of radius of each segment mask
Rs = np.sqrt( (P.sum(axis=1)+ObsArea) / np.pi)

## Normalize rho vector (unitary radius)
rho = rho / Rs[:,np.newaxis,:] * P #final rho vector

# Build an Zernike Influence-function Matrix for all segments (ON-AXIS direction)
alphaId = 0   # only on-axis direction supported...

Zmat = np.zeros((nPx*nPx,Zobj.n_mode,7))
for segId in range(1,8):
    Zobj.reset()
    cutheta = ceo.cuDoubleArray(host_data=theta[segId-1,:,alphaId].reshape(ongs.m,ongs.n))
    curho   = ceo.cuDoubleArray(host_data=  rho[segId-1,:,alphaId].reshape(ongs.m,ongs.n))
    for k in range(Zobj.n_mode):
        Zobj.a[0,k] = 1
        Zobj.update()
        S = Zobj.surface(curho,cutheta).host(shape=(nPx*nPx,1))*P[segId-1,:,alphaId].reshape(-1,1)
        Zmat[:,k,segId-1] = S.flatten()
        Zobj.a[0,k] = 0

print 'Zernike Inflence Function Matrix:'
print Zmat.shape

#Pseudo-inverse of Zmat
invZmat = np.zeros((Zobj.n_mode,nPx*nPx,7))
for segId in range(1,8):
    invZmat[:,:,segId-1] = np.linalg.pinv(Zmat[:,:,segId-1])
print 'inverse of Zernike Influence Function Matrix:'
print invZmat.shape

# Calibrate AO SH WFS - M2 segment TT Interaction Matrix and Reconstructor
TTstroke = 25e-3 #arcsec
gmt.reset()
D_M2_TT = gmt.calibrate(wfs, gs, mirror="M2", mode="segment tip-tilt", stroke=TTstroke*math.pi/180/3600)
R_M2_TT = np.linalg.pinv(D_M2_TT)
print 'AO SH WFS - M2 segment TT Rec:'
print R_M2_TT.shape

# Calibrate AO SH WFS - M2 segment Zernikes Interaction Matrix and Reconstructor
Zstroke = 50e-9 #m rms
D_M2_Z = gmt.calibrate(wfs, gs, mirror="M2", mode="zernike", stroke=Zstroke, first_mode=1)

#remove TT (Z2 & Z3) from IM
D_M2_Z = ((D_M2_Z.reshape(-1,7,n_zern-1))[:,:,2:]).reshape(-1,(n_zern-3)*7)
nzernall = (D_M2_Z.shape)[1]  ## number of zernike DoFs calibrated
print 'AO SH WFS - M2 Segment Zernike IM:'
print D_M2_Z.shape
print 'Condition number: %f'%np.linalg.cond(D_M2_Z)

if VISU == True:
    plt.pcolor(D_M2_Z)
    plt.colorbar()

R_M2_Z = np.linalg.pinv(D_M2_Z)
print 'AO SH WFS - M2 Segment Zernike Rec:'
print R_M2_Z.shape

# Calibrate Segment Piston Sensor Interaction Matrix and Reconstructor
PSstroke = 200e-9 #m
D_M1_PS = gmt.calibrate(ps, gsps, mirror="M1", mode="segment piston", 
                        stroke=PSstroke, segment='edge')

R_M1_PS = np.linalg.pinv(D_M1_PS)
print 'SPS - M1 Segment Piston Rec:'
print R_M1_PS.shape

if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    imm = ax.pcolor(D_M1_PS*1e-6)  #in displaced pixels per micron
    ax.grid()
    ax.set_ylim([0,36])
    ax.set_xticklabels(['S1 $T_z$','S2 $T_z$','S3 $T_z$','S4 $T_z$','S5 $T_z$','S6 $T_z$'], ha='left', fontsize=15, rotation=45, color='b')
    ax.set_yticks([0,12,24,36])
    ax.tick_params(axis='y', labelsize=13)
    ax.text(-0.4,6,'SPS$_1$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    ax.text(-0.4,18,'SPS$_2$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    ax.text(-0.4,30,'SPS$_3$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    fig.colorbar(imm)

# Calibrate FDSP Interaction Matrix and Reconstructor
TTstroke = 50e-3 #arcsec
gmt.reset()
D_FDSP = gmt.calibrate(ps, gsps, mirror="M1", mode="FDSP", stroke=TTstroke*math.pi/180/3600, 
                       segment='edge', agws=wfs, recmat=R_M2_TT)
R_FDSP = np.linalg.pinv(D_FDSP)
print 'SPS - FDSP Rec:'
print R_FDSP.shape

if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(12,4)

    #Rx and Ry are in radians. We want to show IM in microns RMS SURF of tilt
    #We found using DOS that a segment tilt of 47 mas is equivalent to 0.5 microns RMS of tilt on an M1 segment.
    AngleRadians_2_tiltcoeff = 0.5 / (47e-3*math.pi/180/3600) #angle in radians to microns RMS of tilt coeff

    imm = ax.pcolor(D_FDSP/AngleRadians_2_tiltcoeff)  #in displaced pixels per microns RMS of M1 segment tilt
    ax.grid()
    ax.set_ylim([0,36])
    ax.set_xticks(range(12))
    ax.set_xticklabels(['S1 $R_x$','S1 $R_y$','S2 $R_x$','S2 $R_y$','S3 $R_x$','S3 $R_y$',
                        'S4 $R_x$','S4 $R_y$','S5 $R_x$','S5 $R_y$','S6 $R_x$','S6 $R_y$'],
                       ha='left', fontsize=15, rotation=45, color='b')
    ax.set_yticks([0,12,24,36])
    ax.tick_params(axis='y', labelsize=13)
    ax.text(-0.4,6,'SPS$_1$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    ax.text(-0.4,18,'SPS$_2$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    ax.text(-0.4,30,'SPS$_3$', rotation=45, ha='center', va='center', fontsize=15, color='b')
    fig.colorbar(imm)

##### Combine Interaction Matrices of M1 segment piston AND FDSP.
D_PIST = np.concatenate((D_M1_PS, D_FDSP), axis=1)
R_PIST = np.linalg.pinv(D_PIST)
print 'Merged SPS - PISTON Rec:'
print R_PIST.shape

if VISU == True:
    plt.pcolor(D_PIST)
    plt.colorbar()

##### Reset before starting
gs.reset()
gsps.reset()
ongs.reset()
gmt.reset()

##### Apply a known Tilt to a particular segment on M1
M1RotVec = np.array([  #arcsec
            [0,0,0], #[200e-3,0,0] ,
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]]) * math.pi/180/3600   
#M1RotVec = myRotVec
##### Apply a known segment piston/translation to a particular segment on M1
M1TrVec = np.array([  # meters surf
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]])

for idx in range(7): gmt.M1.update(origin=M1TrVec[idx,:].tolist(), euler_angles=M1RotVec[idx,:].tolist(), idx=idx+1)

##### Apply a known Tilt to a particular segment on M2
M2RotVec = np.array([  #arcsec
            [0,0,0] ,
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]]) * math.pi/180/3600   

##### Apply a known segment piston/translation to a particular segment on M2
M2TrVec = np.array([  # meters surf
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]])

for idx in range(7): gmt.M2.update(origin=M2TrVec[idx,:].tolist(), euler_angles=M2RotVec[idx,:].tolist(), idx=idx+1)    

# Apply a Zernike vector to a particular segment on M1
mysegId = 1
a_M1 = np.zeros(n_zern)   #zernike coeffs, from piston Z1 to n_zern
a_M1[10] = 500e-9      # m RMS surf
gmt.M1.zernike.a[mysegId-1,:] = a_M1
#for mmmm in range(6): gmt.M1.zernike.a[mmmm,:] = a_M1
gmt.M1.zernike.update()

if VISU == True:
    gmt.propagate(ongs)
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    imm = ax.imshow(ongs.phase.host(units='micron'), interpolation='None',cmap='RdYlBu',origin='lower')#, vmin=-8, vmax=8)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    clb = fig.colorbar(imm, ax=ax, fraction=0.012, pad=0.03,format="%.1f")
    clb.set_label('$\mu m$ WF', fontsize=25)
    clb.ax.tick_params(labelsize=15)
    ongs.reset()

##### Close the loop !!!!!
niter = 5
TTniter = 5

if VISU == True:
    f, (ax1,ax2) = plt.subplots(2,1)
    f.set_size_inches(15,10)

rmsval = np.zeros(niter*TTniter)
myTTest1 = np.zeros((7,2))
M2TTresiter = np.zeros((7,2,niter*TTniter))
myPSest1 = np.zeros(6)
myFDSPest1 = np.zeros((6,2))  #6x M1 segment tip-tilt
M1TTresiter = np.zeros((7,2,niter*TTniter))
M1PSresiter = np.zeros((7,niter*TTniter))
a_M2 = np.zeros(nzernall)
a_M2_iter = np.zeros((nzernall,niter*TTniter))

for ii in range(niter):
    for jj in range(TTniter):
        ongs.reset()
        gs.reset()
        gmt.propagate(ongs)
        gmt.propagate(gs)
        if VISU == False and jj == TTniter-1:
            gsps.reset()
            gmt.propagate(gsps)
        rmsval[ii*TTniter+jj] = ongs.wavefront.rms()        
        
        #---- visualization
        if VISU == True:
            gsps.reset()
            gmt.propagate(gsps)
            if ii*TTniter+jj > 0: clb.remove()
            h = ax1.imshow(ongs.phase.host(units='micron')-ph_fda_on,interpolation='None')
            ax1.set_title(ii*TTniter+jj, fontsize=20)
            clb = f.colorbar(h, ax=ax1, shrink=0.8)
            clb.set_label('um')
            if ii*TTniter+jj > 0: clb2.remove()
            h2 = ax2.imshow(gsps.phase.host(units='micron').T-ph_fda,interpolation='None')
            ax2.set_title(ii*TTniter+jj, fontsize=20)
            clb2 = f.colorbar(h2, ax=ax2, shrink=0.8)
            clb2.set_label('um')
            IPython.display.clear_output(wait=True)
            IPython.display.display(f)
        
        #---- SH WFS measurement
        wfs.reset()
        wfs.analyze(gs)
        slopevec = wfs.valid_slopes.host().ravel()
        
        #---- segment TT correction
        myTTest1 += np.dot(R_M2_TT, slopevec).reshape((7,2))
        M2RotCor = np.zeros((7,3))
        M2RotCor[:,0:2] = myTTest1
        M2RotVec1 = M2RotVec - M2RotCor
        M2TTresiter[:,:,ii*TTniter+jj] = M2RotVec1[:,0:2]
        if ii*TTniter+jj >= TTniter:
            M1TTresiter[:,:,ii*TTniter+jj] = M1RotVec1[:,0:2]
            M1PSresiter[:,ii*TTniter+jj] = M1TrVec1[:,2]
        else:
            M1TTresiter[:,:,ii*TTniter+jj] = M1RotVec[:,0:2]
            M1PSresiter[:,ii*TTniter+jj] = M1TrVec[:,2]
        for idx in range(7): gmt.M2.update(origin=M2TrVec[idx,:].tolist(), euler_angles=M2RotVec1[idx,:].tolist(), idx=idx+1)
    
        #---- segment Zernikes correction (on M2)
        a_M2 += np.dot(R_M2_Z, slopevec) 
        a_M2_iter[:,ii*TTniter+jj] = a_M2
        atemp = a_M2.reshape((7,-1))
        gmt.M2.zernike.a[:,3:] = -atemp
        gmt.M2.zernike.update()

    ##-- FDSP and segment piston correction    
    
    ps.reset()
    PISTmeas = ps.piston(gsps, segment='edge') - SPSmeas_ref
    myPISTest1 = np.dot(R_PIST, PISTmeas.ravel())
    #--- segment piston
    myPSest1 += myPISTest1[0:6]
    M1TrCor = np.zeros((7,3))
    M1TrCor[0:6,2] = myPSest1
    M1TrVec1 = M1TrVec - M1TrCor
    #--- FDSP
    myFDSPest1 += myPISTest1[6:].reshape((6,2))
    M1RotCor = np.zeros((7,3))
    M1RotCor[0:6,0:2] = myFDSPest1
    M1RotVec1 = M1RotVec - M1RotCor
    for idx in range(7): gmt.M1.update(origin=M1TrVec1[idx,:].tolist(), euler_angles=M1RotVec1[idx,:].tolist(), idx=idx+1)
    
      
    """
    ##-- FDSP correction only
    ps.reset()
    FDSPmeas = ps.piston(gs, segment='edge') - SPSmeas_ref
    myFDSPest1 += np.dot(R_FDSP, FDSPmeas.ravel()).reshape((6,2))
    M1RotCor = np.zeros((7,3))
    M1RotCor[0:6,0:2] = myFDSPest1
    M1RotVec1 = M1RotVec - M1RotCor
    for idx in range(7): gmt.M1.update(origin=M1TrVec[idx,:].tolist(), euler_angles=M1RotVec1[idx,:].tolist(), idx=idx+1)
      
    ##-- segment Piston correction only
    ps.reset()
    SPSmeas = ps.piston(gsps, segment='edge') - SPSmeas_ref
    myPSest1 += np.dot(R_M1_PS, SPSmeas.ravel())
    M1TrCor = np.zeros((7,3))
    M1TrCor[0:6,2] = myPSest1
    M1TrVec1 = M1TrVec - M1TrCor
    for idx in range(7): gmt.M1.update(origin=M1TrVec1[idx,:].tolist(), euler_angles=M1RotVec[idx,:].tolist(), idx=idx+1)
    """
ongs.reset()
gsps.reset()
gmt.propagate(ongs)
gmt.propagate(gsps)
Wres = np.rollaxis (ongs.wavefront.phase.host(units='nm', shape=(1,ongs.N_SRC,ongs.n*ongs.m)),1,3)
on_rms = ongs.wavefront.rms()*1e9

#---- visualization
if VISU == True:
    clb.remove()
    h = ax1.imshow(ongs.phase.host(units='micron')-ph_fda_on,interpolation='None')
    clb = f.colorbar(h, ax=ax1, shrink=0.8)
    clb.set_label('um')
    clb2.remove()
    h2 = ax2.imshow(gsps.phase.host(units='micron').T-ph_fda,interpolation='None')
    clb2 = f.colorbar(h2, ax=ax2, shrink=0.8)
    clb2.set_label('um')    
    IPython.display.clear_output(wait=True)
    IPython.display.display(f)
    plt.close()

plt.pcolor(SPSmeas.T)
plt.colorbar()

show_sps_imagettes()

### Show final on-axis residual WF
VISU=True
if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(20,5)
    imm = ax.imshow(ongs.phase.host(units='nm')-ph_fda_on*1e3, interpolation='None',cmap='RdYlBu', origin='lower')#, vmin=-1.5, vmax=1.5)
    ax.set_title('on-axis WF')
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    clb = fig.colorbar(imm, ax=ax, fraction=0.012, pad=0.03,format="%.1f")
    clb.set_label('nm WF', fontsize=25)
    clb.ax.tick_params(labelsize=15)

#### Compute residual WFE in the Field
sfgs.reset()
gmt.propagate(sfgs)
sf_rms = sfgs.wavefront.rms()*1e9

if VISU == True:
    fig,ax = plt.subplots()
    fig.set_size_inches((5,4))
    contp = ax.contourf(sfx, sfy, sf_rms.reshape(len(sfx),-1))
    #contp = ax.imshow(sf_rms.reshape(len(sfx),-1), extent=[-45, 45, -45, 45], origin='lower', interpolation='bilinear')
    clb = fig.colorbar(contp, ax=ax)
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.set_xlabel('field angle [arcsec]', fontsize=15)
    clb.set_label('nm WF RMS', fontsize=15)
    clb.ax.tick_params(labelsize=12)

print '--> WF RMS on-axis: %3.2f nm wf RMS'%on_rms
print '--> max WF RMS at the edge of the field: %3.2f nm WF RMS'%np.max(sf_rms)

#### on-axis WFE vs. iteration number 

# final wf rms value over each segment
WF = (ongs.phase.host(units='micron')-ph_fda_on).reshape(nPx*nPx)
print 'Final WFE over each segment [microns RMS]:'
print np.array_str(segment_rms(WF,P).T*1e3, precision=3, suppress_small=True)
VISU=True
if VISU == True:
    fig, ax = plt.subplots()
    fig.set_size_inches(15,5)
    ax.semilogy(rmsval*1e9, '-+')
    ax.grid()
    ax.set_xlabel('iteration', fontsize=20)
    ax.set_ylabel('nm WF rms', fontsize=20)
    ax.tick_params(labelsize=15)

#### Residual segment piston analysis

print 'Final M1 final piston (Tz) values [nm WF]:'
print np.array_str(gmt.M1.motion_CS.origin[:,2]*1e9*2, precision=3, suppress_small=True)
print '-----'
print 'Final M2 final piston (Tz) values [nm WF]:'
print np.array_str(gmt.M2.motion_CS.origin[:,2]*1e9*2, precision=3, suppress_small=True)

if VISU == True:
    f1, ax1 = plt.subplots()
    f1.set_size_inches(7,5)
    ax1.plot(M1PSresiter.T *1e9*2, label='S')
    ax1.grid()
    #ax1.set_title('Tz', size='x-large')
    ax1.set_xlabel('iteration', size='xx-large')
    ax1.set_ylabel('M1 segment piston [nm WF]', size='xx-large')
    ax1.tick_params(labelsize=18)
    #ax1.legend()
    initpos = 11
    deltapos = 2
    plt.text(initpos,11,'S1', color='b', ha='center', fontsize=18)
    plt.text(initpos+deltapos,11,'S2', color='g', ha='center', fontsize=18)
    plt.text(initpos+2*deltapos,11,'S3', color='r', ha='center', fontsize=18)
    plt.text(initpos+3*deltapos,11,'S4', color='c', ha='center', fontsize=18)
    plt.text(initpos+4*deltapos,11,'S5', color='m', ha='center', fontsize=18)
    plt.text(initpos+5*deltapos,11,'S6', color='y', ha='center', fontsize=18)
    plt.text(initpos+6*deltapos,11,'S7', color='k', ha='center', fontsize=18)

#### Residual M1 / M2 segment Tip-tilt analysis

print '------'
print 'Final M2 final TT (Rx, Ry) values [mas]:'
print np.array_str(gmt.M2.motion_CS.euler_angles[:,0]*ceo.constants.RAD2MAS, precision=3, suppress_small=True)
print np.array_str(gmt.M2.motion_CS.euler_angles[:,1]*ceo.constants.RAD2MAS, precision=3, suppress_small=True)
print '------'
print 'Final M1 final TT (Rx, Ry) values [mas]:'
print np.array_str(gmt.M1.motion_CS.euler_angles[:,0]*ceo.constants.RAD2MAS, precision=3, suppress_small=True)
print np.array_str(gmt.M1.motion_CS.euler_angles[:,1]*ceo.constants.RAD2MAS, precision=3, suppress_small=True)

if VISU == True:
    f1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.axis('auto')
    ax2.axis('auto')
    ax3.axis('auto')
    ax4.axis('auto')
    f1.set_size_inches(14,10)

    ax1.plot(M2TTresiter[:,0,:].T * ceo.constants.RAD2MAS)
    ax1.grid()
    ax1.set_title('Rx', size='x-large')
    ax1.set_xlabel('iteration', size='x-large')
    ax1.set_ylabel('M2 segment y-tilt [mas]', size='x-large')
    ax1.tick_params(labelsize=13)

    ax2.plot(M2TTresiter[:,1,:].T *ceo.constants.RAD2MAS)
    ax2.grid()
    ax2.set_title('Ry', size='x-large')
    ax2.set_xlabel('iteration', size='x-large')
    ax2.set_ylabel('M2 segment x-tilt [mas]', size='x-large')
    ax2.tick_params(labelsize=13)

    ax3.plot(M1TTresiter[:,0,:].T *ceo.constants.RAD2MAS*8)
    ax3.grid()
    #ax3.set_title('Rx', size='x-large')
    ax3.set_xlabel('iteration',size='x-large')
    ax3.set_ylabel('M1 segment y-tilt [mas]', size='x-large')
    ax3.tick_params(labelsize=13)

    ax4.plot(M1TTresiter[:,1,:].T *ceo.constants.RAD2MAS*8)
    ax4.grid()
    #ax4.set_title('Ry', size='x-large')
    ax4.set_xlabel('iteration', size='x-large')
    ax4.set_ylabel('M1 segment x-tilt [mas]', size='x-large')
    ax4.tick_params(labelsize=13)

#### M1/M2 segment tilt correlation 
#### Basically, if points lie on a line it means M1 and M2 tilts are compensating each other.
if VISU == True:
    f1, (ax1,ax2) = plt.subplots(1,2)
    f1.set_size_inches(14,4)

    ax1.plot(gmt.M2.motion_CS.euler_angles[:,0]*ceo.constants.RAD2MAS,
         gmt.M1.motion_CS.euler_angles[:,0]*ceo.constants.RAD2MAS, '-o')
    ax1.grid()
    ax1.set_title('Rx', size='x-large')
    ax1.set_xlabel('M2 segment tilt', size='large')
    ax1.set_ylabel('M1 segment tilt', size='large')
    ax2.plot(gmt.M2.motion_CS.euler_angles[:,1]*ceo.constants.RAD2MAS,
         gmt.M1.motion_CS.euler_angles[:,1]*ceo.constants.RAD2MAS, '-o')
    ax2.grid()
    ax2.set_title('Ry', size='x-large')
    ax2.set_xlabel('M2 segment tilt', size='large')
    ax2.set_ylabel('M1 segment tilt', size='large')

print '------'
print 'Final M1 Zernike (from Z4) coeffs [nm]:'
print np.array_str(a_M1[3:]*1e9, precision=3, suppress_small=True)  # zernike coeffs [z4 onwards]
print 'Final M2 Zernike (from Z4) coeffs [nm]:'
print np.array_str((1e9*a_M2.reshape((7,-1)))[mysegId-1,:], precision=3, suppress_small=True)  #Z4 onwards

#### Residual M2 segment Zernikes
if VISU == True:
    thisSeg = 1  # show coeffs vs iteration for this segment only.
    my_a_M2 = (a_M2_iter.reshape((7,-1,niter*TTniter)))[thisSeg-1,:,:]

    f1, ax1 = plt.subplots()
    f1.set_size_inches(7,5)
    ax1.plot(my_a_M2.T*1e9, '-o')
    ax1.grid()
    ax1.set_title('M2 segment #%d'%thisSeg, size='x-large')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('Zcoeff [nm RMS SURF]', size='large')

#Project final residual WF onto segment Zernike modes
arec = np.zeros((Zobj.n_mode, 7))
for segId in range(1,8):
    arec[:,segId-1] = np.dot(invZmat[:,:,segId-1], (Wres-Wref)[:,:,alphaId].reshape(-1))

print 'Zernike Coeffs (Z1 onwards):'
print '============================'
for segId in range(1,8):
    print 'segment #%d: -----------'%segId
    print np.array_str(arec[:,segId-1],precision=3,suppress_small=True)

rmsval*1e9

gmt.M1.zernike.a



