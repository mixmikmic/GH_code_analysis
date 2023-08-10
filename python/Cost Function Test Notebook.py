from lib import Cost_clean as C
from lib import houseLoad as house
from lib import evLoad 
import numpy as np
COST=C.Cost('db/Tariff_Rate_New.db','db/Household_15mins.db')
ev = evLoad.EV('db/ev_model.db')

#########Case 1############
'''
Cost including EV Load, water pump load and HVAC load
Input Utility Name: Duke Energy Florida
Input Rate Name: Residential TOU
TOU Type Rate(TOU 1,0,0)
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='Duke Energy Florida'
Rate_Name='Residential TOU'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,1,1]
Charging_Outside=1 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)

print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 2############
'''
Cost doesn't include EV Load, water pump load and HVAC load
Input Utility Name: Duke Energy Florida
Input Rate Name: Residential TOU
TOU Type Rate(TOU 1,0,0)
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='Duke Energy Florida'
Rate_Name='Residential TOU'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=1 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 3############
'''
Input Utility Name: PG&E
Input Rate Name: ETOUA
TOU and Tier Type Rate(TOU 1,1,0)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='PG&E'
Rate_Name='ETOUA'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=1 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 4############
'''
Input Utility Name: Duke Energy North Carolina
Input Rate Name: Residential TOU
TOU and Tier Type Rate(TOU 1,0,1)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(30, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='Duke Energy North Carolina'
Rate_Name='Residential TOU'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 5############
'''
Input Utility Name: PG&E
Input Rate Name: ETOUB
TOU and Tier Type Rate(TOU 1,0,0)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='PG&E'
Rate_Name='ETOUB'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 6############
'''
Input Utility Name: PG&E
Input Rate Name: E1
Fixed Tier Type Rate
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='PG&E'
Rate_Name='E1'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 7############
'''
Input Utility Name: Duke Energy North Carolina
Input Rate Name: Residential Service Rate
Fixed Tier Type Rate(Only 1 Tier)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(50, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='Duke Energy North Carolina'
Rate_Name='Residential Service Rate'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,1,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 8############
'''
Input Utility Name: Duke Energy Indiana
Input Rate Name: Residential and Farm Service
Fixed Tier Type Rate(3 Tier)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(20, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='Duke Energy Indiana'
Rate_Name='Residential and Farm Service'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

#########Case 8############
'''
Input Utility Name: Duke Energy Indiana
Input Rate Name: Residential and Farm Service
Fixed Tier Type Rate(3 Tier)
Cost including EV Load, water pump load and HVAC load
Input Cost:200

'''
#EV Load Profile Generation
Cost=200
data = ev.get_load_profile(20, 'Nissan', 'Leaf', 2012, 0)
EV_Load=np.round(data['load_profile'], 2)
######Input Arguments############
Utility_Name='PG&E'
Rate_Name='EV'
N_room=2;N_day=3;N_night=0;#Ls_App=[];
No_EV=5;
Ls_App=[1,0,1,1,0,0]
Charging_Outside=0 #EV charging outside the home
################################

#Step 1:Calculating the Cost Reduction Based on the Appliances and EV_load Input.
EV_Cost=COST.Get_EV_Def_Cost(Charging_Outside,Utility_Name,Rate_Name,EV_Load,Ls_App,No_EV,Cost)
print 'The Cost Portion of EV_load,Water pump load and HVAC load is:  '+str(EV_Cost)

#########################################################
#Step 2:Preding the electricity consumption w or w/o Cost

#Case 1: Cost not Given(Cost=0):
Consumption_Case1=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost=0,No_EV=5)
print 'Consumption prediction w/o input cost data : '+str(Consumption_Case1)
#Case 2: Cost Given:
Consumption_Case2=COST.Get_Monthly_Consumption(Charging_Outside,Utility_Name,Rate_Name,N_room, N_day,N_night,Ls_App,EV_Load,Cost,No_EV=5)
print 'Consumption prediction w input cost data : '+str(Consumption_Case2)

############################################################
#Step 3:Get Household Profile Based on the input consumption
Cust_Total_Profile, Cust_Profile1, Deferred_Matrix = house.get_household_load_profile(Consumption_Case2,N_room, N_day,N_night,Ls_App,connection_time=[0, 0, 0])

#############################
#Step 4: Calculating the Cost
Plan_COST=COST.Get_Cost(Utility_Name,Cust_Total_Profile,Cust_Profile1, Deferred_Matrix,EV_Load,Charging_Outside,No_EV=5,No_Def=2)
print '#########Results############################'
print 'Plan Name:'+str(Plan_COST['Plan_Name'])
print 'Total Summer Cost:'+str(Plan_COST['Total_S'])
print 'Total Winter Cost:'+str(Plan_COST['Total_W'])

