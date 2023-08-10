liters_per_m3 = 1000.0
gallons_per_m3 = 264.17

V_lpm = (1.0*liters_per_m3/gallons_per_m3)/30.0      # volume of gasline in liters/mile
m_kg = 0.74*V_lpm                                    # mass of gasoline in kg/mile
m_grams = m_kg*1000.0                                # mass of gasoline in grams/mile
n_octane = m_grams/114.0                             # moles of gasoline in gmol/mile
n_co2 = 8.0*n_octane                                 # modles of CO2 in gmol/mile
m_co2 = 44.0*n_co2                                   # mass of CO2 in grams/mile

print("Gasoline consumed per mile = ", round(m_grams,1), "g/mile")
print("Gram moles of octane per mile = ", round(n_octane,3) ,"gmol/mile")
print("CO2 Production =", round(m_co2,1), "g/mile")

grams_per_lb = 453.593

w_kwh = 0.367                     # kwh per mile
q_btu = (w_kwh/0.8)*10400.0       # natural gas per mile

print("Thermal energy requirement =",round(q_btu,2),"BTU per mile driven")

m_co2_lb = 117.0*q_btu/1.0e6           # mass CO2 lb/mile
m_co2_grams = m_co2_lb*grams_per_lb    # mass CO2 grams/mile

print("CO2 Production =", round(m_co2_grams,2), "grams per mile")



