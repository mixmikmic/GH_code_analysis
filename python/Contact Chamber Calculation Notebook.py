import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from aide_design import physchem as pc

from aide_design.units import unit_registry as u

from aide_design import utility as ut

from aide_design.play import*
from aide_design import floc_model as floc
from pytexit import py2tex

import math

from scipy import constants, interpolate

#see numpy cheat sheet https://www.dataquest.io/blog/images/cheat-sheets/numpy-cheat-sheet.pdf
#The numpy import is needed because it is renamed here as np.
import numpy as np

#Pandas is used to import data from spreadsheets
import pandas as pd
 
import matplotlib.pyplot as plt

# sys and os give us access to operating system directory paths and to sys paths.
import sys, os

PipeArea = (np.pi*(0.0127**2))*(u.m**2)
Upflow_Velocity = 0.002 *(u.m/u.s)
SedTube_Flow = (PipeArea * Upflow_Velocity).to(u.mL/u.s)
print('The flow through the sedimentation tube is ' +ut.sig(SedTube_Flow, 4)+ '.')

WaterPump_rpm = ((SedTube_Flow/0.8)*(60*(u.s))).magnitude
print('The required RPMs for the water pump to achive a flow rate of 1.52 mL/s is ' +ut.sig(WaterPump_rpm,4)+' RPM.')

Flow_plant = 180*(u.mL/u.min)
Q_coagulant = 5 * (u.mL/u.min)
Coagulant_Concentration = 0.1418 * (u.g/u.L)
Concentration_Plant = ((Q_coagulant * Coagulant_Concentration)/Flow_plant).to(u.mg/u.L)
print('The concentration of coagulant throughout the system is ' +ut.sig(Concentration_Plant,4)+ '.')

#15RPM for Coagulant Pump, 76 RPM Water
Coag_Conc = 0.2836 * (u.g/u.L)
Q_plant = 180 * (u.mL/u.min)
Q_coag = 5 * (u.mL/u.min)
C_plant= ((Q_coag*Coag_Conc)/Q_plant).to(u.mg/u.L)
print('The coagulant concentration throughout the system in this trial is ' +ut.sig(C_plant,4)+ '.')

YB_RPM_Conversion = 0.149
MediumSpeed = 40
HighSpeed = 60
MediumFlow = (YB_RPM_Conversion * MediumSpeed)*(u.mL)
HighFlow = (YB_RPM_Conversion * HighSpeed)*(u.mL)
print('The flow rate at this pump speed is ' +ut.sig(MediumFlow,3)+' per revolution')
print('The flow rate at this pump speed is ' +ut.sig(HighFlow,3)+' per revolution')

Sixteen_RPM_Conversion = 0.8 #mL/revolution
WaterSpeed = 76 #revolutions per minute
Q_Water = (WaterSpeed * Sixteen_RPM_Conversion)*(u.mL/u.min)
print('The flow of water into the system is ' +ut.sig(Q_Water,4)+ '.')

Coag_Speed = 10 #RPM 
Q_Stock = (YB_RPM_Conversion * Coag_Speed)*(u.mL/u.min)
print('The flow of coagulant into the system is ' +ut.sig(Q_Stock,4)+ '.')

Clay_Speed = 16 #Approximate pump speed - variable
Q_Clay = (YB_RPM_Conversion * Clay_Speed)*(u.mL/u.min)
print('The flow of clay solution into the system is ' +ut.sig(Q_Clay,4)+ '.')

Plant_Flow = Q_Water + Q_Stock + Q_Clay
print('The total flow rate through the system is ' +ut.sig(Plant_Flow,4)+ '.')

#15RPM for Coagulant Pump, 76 RPM Water
StockCoag_Conc = (0.2836/4) * (u.g/u.L) #Coaglant flowing from stock
TotalSystem_Flow = 64.67 * (u.mL/u.min) #Raw Water + Clay + Coagulant
Q_coag = 1.49* (u.mL/u.min)
C_plant= ((Q_coag*StockCoag_Conc)/TotalSystem_Flow).to(u.mg/u.L)
print('The coagulant concentration throughout the system in this trial is ' +ut.sig(C_plant,4)+ '.')



