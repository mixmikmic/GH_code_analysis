from aide_design.play import*
import math
from pytexit import py2tex

def UASBSize(diam, height):
    """Takes the inputs of diameter and height. The bottom of the UASB is sloped
    at 60 degrees with a 3 inch space across the bottom of the UASB. Assumes that half the reactor 
    contains the settled bed, which is used for the HRT. Returns five outputs: 1. height of the sloped
    sides of the bottom geometry, 2. volume of sludge in the reactor, 3. flow rate,
    4. number of people served with graywater, 5. number of people served with blackwater. 
    """
    
    WW_gen = 3 * u.mL/u.s        #Wastewater generated per person, rule of thumb from Monroe
    WW_gen_bw = 0.6 * u.mL/u.s   #Assumes 20% of mixed wastewater
    HRT = 4 * u.hr               #Hydraulic Residence Time, determined from lab scale tests
    
    centerspace = 3 * u.inch     #Center space allows for a 3 inch pipe across the bottom
    phi = math.atan((diam/2)/(centerspace/2))
    angle = 60 * u.deg           #Angle of the sloped bottom
    
    height_cyl_hoof = diam/2 * np.tan(angle)    #Hoof is if 
    height_cyl_wedge = height_cyl_hoof - ((centerspace/2) * math.tan(angle))
    vol_cyl_wedge = height_cyl_wedge * (diam/2)**2 / 3 * ((
        3*math.sin(phi) - 3*phi*math.cos(phi) - math.sin(phi)**3)/(1-math.cos(phi)))
    vol_reactor = (math.pi * (diam / 2)**2 * height / 2) - (2 * vol_cyl_wedge)
    
    flow = vol_reactor / HRT
    people_served = int(flow / WW_gen)       #People served per reactor
    people_served_BW = int(flow / WW_gen_bw) #People served per reactor treating only blackwater
    
    output = [height_cyl_wedge.to(u.m), vol_reactor.to(u.L), flow.to(u.L/u.s), people_served, people_served_BW]
    
    print("The height of the bottom geometry is",height_cyl_wedge.to(u.m))
    print('The volume of the sludge in the reactor is', vol_reactor.to(u.L))
    print('The flow rate of the reactor is', flow.to(u.L/u.s))
    print('The number of people served by this reactor is', people_served)
    print('The number of people served by this reactor if only blackwater is treated is', people_served_BW)
    return output

UASB = UASBSize(3*u.ft, 7*u.ft)

# Design Parameters
height_blanket = 3.5 * u.ft 
plate_space = 2.5 * u.cm
angle = 60 * u.deg
thickness_sed_plate = 2 * u.mm
flow = UASB[2]

# Assumptions
diam_sludge_weir = 6 * u.inch
freespace = 12 * u.inch
water_elevation = 6.5 * u.ft  ## figure out from previous reports

diam_tube = np.array([8,10]) * u.inch

velocity_tube_alpha = (flow/(pc.area_circle(diam_tube))).to(u.mm/u.s)

print(velocity_tube_alpha.magnitude,velocity_tube_alpha.units )

# velocity_plate = np.sin(angle) * velocity_tube_alpha
# print(velocity_plate.to(u.mm/u.s))

# projected_area = (((length_tube_settler * np.cos(angle)
#                   ) + (plate_space/np.sin(angle))) * diam_tube)

# velocity_capture = ((velocity_plate * pc.area_circle(diam_tube))/(np.sin(angle))
#                    )/projected_area

# print(velocity_capture.to(u.mm/u.s))


height_tube_settler = height_blanket + diam_sludge_weir + freespace + 0.5*diam_tube
print(height_tube_settler.to(u.inch))

length_tube_settler_vertical = water_elevation - height_tube_settler
print(length_tube_settler_vertical.to(u.inch))

length_tube_settler = np.sin(angle) * length_tube_settler_vertical
print(length_tube_settler.to(u.inch))





