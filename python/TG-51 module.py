"""A script to calculate TG-51 dose using pylinac functions and following the TG-51 photon form"""
from pylinac import tg51


ENERGY = 6
TEMP = 22.1
PRESS = 755.0
CHAMBER = '30013'  # PTW
P_ELEC = 1.000
ND_w = 5.443  # Gy/nC
MU = 200
CLINICAL_PDD = 66.5

# Section 4 (beam quality)
# since energy is 6MV, PDDx == PDD
pdd10x = 66.4

# Section 5 (kQ)
kq = tg51.kq(model=CHAMBER, pddx=pdd10x)

# Section 6 (Temp/Press)
p_tp = tg51.p_tp(temp=TEMP, press=PRESS)

# Section 7 (polarity)
m_raw = m_neg = (25.66, 25.67, 25.66)
m_pos = (25.67, 25.67, 25.68)
p_pol = tg51.p_pol(m_reference=m_neg, m_opposite=m_pos)

# Section 8 (ionization)
m_low = (25.64, 25.64, 25.65)
p_ion = tg51.p_ion(volt_high=300, volt_low=150, m_high=m_raw, m_low=m_low)

# Section 9 (M corrected)
m_corr = tg51.m_corrected(p_ion=p_ion, p_tp=p_tp, p_elec=P_ELEC, p_pol=p_pol, m_raw=m_raw)

# Section 10 (dose to water @ 10cm)
dose_10 = m_corr*kq*ND_w
dose_10_per_mu = dose_10 / MU

# Section 11 (dose/MU to water @ dmax)
dose_ddmax = dose_10_per_mu / CLINICAL_PDD

# Done!
print(dose_ddmax)

ENERGY = 6
TEMP = 22.1
PRESS = 755.0
CHAMBER = '30013'  # PTW
P_ELEC = 1.000
ND_w = 5.443  # Gy/nC
MU = 200
CLINICAL_PDD = 66.5

tg51_6x = tg51.TG51Photon(temp=TEMP, press=PRESS, model=CHAMBER,
                          n_dw=ND_w, p_elec=P_ELEC,
                          measured_pdd=66.4, lead_foil=None,
                          clinical_pdd=66.5, energy=ENERGY,
                          volt_high=-300, volt_low=-150,
                          m_raw=(25.65, 25.66, 25.65),
                          m_opp=(25.64, 25.65, 25.65),
                          m_low=(25.64, 25.63, 25.63),
                          mu=MU, tissue_correction=1.0)

# Done!
print(tg51_6x.dose_mu_dmax)

# examine other parameters
tg51_6x.pddx
tg51_6x.kq
tg51_6x.p_ion

# change readings if you adjust output
tg51_6x.m_raw = (25.44, 25.44, 25.43)
# print new dose value
print(tg51_6x.dose_mu_dmax)



