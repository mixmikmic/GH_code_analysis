from sympy import *
init_printing()

Id, W,L, VDS, VGS, Vnth, VnA, nTechConst=symbols("I_D, W, L, V_DS, V_GS, V_thn, V_An, k_n'")
Id, W,L, VDS, VGS, Vnth, VnA, nTechConst

NLinFull=nTechConst*(W/L)*((VGS-Vnth)*VDS-VDS**2/2)*(1+VDS/VnA)
NLinFull

NLinSimp=nTechConst*(W/L)*((VGS-Vnth)*VDS-VDS**2/2)
NLinSimp

NSatFull=nTechConst/2*W/L*(VGS-Vnth)**2 *(1+VDS/VnA)
NSatFull

NSatSimp=nTechConst/2*W/L*(VGS-Vnth)**2
NSatSimp

NIdFull=Piecewise(
    (0, VGS<Vnth),
    (NLinFull, And(VGS>=Vnth, VDS<VGS-Vnth)),
    (NSatFull, And(VGS>=Vnth, VDS>=VGS-Vnth))
)
NIdFull

NIdSimp=Piecewise(
    (0, VGS<Vnth),
    (NLinSimp, And(VGS>=Vnth, VDS<VGS-Vnth)),
    (NSatSimp, And(VGS>=Vnth, VDS>=VGS-Vnth))
)
NIdSimp

NCutoffCond=VGS<Vnth; NCutoffCond

NLinearCond=VDS<VGS-Vnth; NLinearCond

NSatCond=VDS>=VGS-Vnth; NSatCond

NLinFull.subs({})

class NMOS():
    def __init__():
        pass
    
    @staticmethod
    def LinFull(subs={}):
        return NLinFull.subs(subs)
    
    @staticmethod
    def LinSimp(subs={}):
        return NLinSimp.subs(subs)
    
    @staticmethod
    def SatFull(subs={}):
        return NSatFull.subs(subs)
    
    @staticmethod
    def IdFull(subs={}):
        return NIdFull.subs(subs)
    
    @staticmethod
    def IdSimp(subs={}):
        return NIdSimp.subs(subs)
    
    @staticmethod
    def CutoffCond(subs={}):
        return NCutoffCond.subs(subs)
    
    @staticmethod
    def LinearCond(subs={}):
        return NLinearCond.subs(subs)
    
    @staticmethod
    def SatCond(subs={}):
        return NSatCond.subs(subs)

VSD, VSG, Vpth, VpA, pTechConst=symbols("V_SD, V_SG, V_thp, V_Ap, k_p'")
VSD, VSG, Vpth, VpA, pTechConst

PLinFull=pTechConst*(W/L)*((VSG-abs(Vpth))*VSD-VSD**2/2)*(1+VSD/abs(VpA))
PLinFull

PLinSimp=pTechConst*(W/L)*((VSG-abs(Vpth))*VSD-VSD**2/2)
PLinSimp

PSatFull=pTechConst/2*W/L*(VSG-abs(Vpth))**2 *(1+VSD/abs(VpA))
PSatFull

PSatSimp=pTechConst/2*W/L*(VSG-abs(Vpth))**2
PSatSimp

PIdFull=Piecewise(
    (0, VSG<abs(Vpth)),
    (PLinFull, And(VSG>=abs(Vpth), VSD<VSG-abs(Vpth))),
    (PSatFull, And(VSG>=abs(Vpth), VSD>=VSG-abs(Vpth)))
)
PIdFull

PIdSimp=Piecewise(
    (0, VSG<abs(Vpth)),
    (PLinSimp, And(VSG>=abs(Vpth), VSD<VSG-abs(Vpth))),
    (PSatSimp, And(VSG>=abs(Vpth), VSD>=VSG-abs(Vpth)))
)
PIdSimp

PCutoffCond=VSG<abs(Vpth); PCutoffCond

PLinearCond=VSD<VSG-abs(Vpth); PLinearCond

PSatCond=VSD>=VSG-abs(Vpth); PSatCond

class PMOS():
    def __init__():
        pass
    
    @staticmethod
    def LinFull(subs={}):
        return PLinFull.subs(subs)
    
    @staticmethod
    def LinSimp(subs={}):
        return PLinSimp.subs(subs)
    
    @staticmethod
    def SatFull(subs={}):
        return PSatFull.subs(subs)
    
    @staticmethod
    def IdFull(subs={}):
        return PIdFull.subs(subs)
    
    @staticmethod
    def IdSimp(subs={}):
        return PIdSimp.subs(subs)
    
    @staticmethod
    def CutoffCond(subs={}):
        return PCutoffCond.subs(subs)
    
    @staticmethod
    def LinearCond(subs={}):
        return PLinearCond.subs(subs)
    
    @staticmethod
    def SatCond(subs={}):
        return PSatCond.subs(subs)



