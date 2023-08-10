from taxcalc import *
from functions2 import *
import copy
import pandas as pd
import numpy as np

# Total benefits from cps
cps = pd.read_csv('cps_benefit.csv')
cps['tot_benefits'] = cps['MedicareX'] + cps['MEDICAID'] + cps['SS'] + cps['SSI'] + cps['SNAP'] + cps['VB']
cps_rev = (cps['tot_benefits'] * cps['s006']).sum()

# Total benefits from other programs
other_programs = pd.read_csv('benefitprograms.csv')
other_programs['Cost'] *= 1000000
other_rev = other_programs['Cost'].sum()

# Allocate benefits from other programs to individual
cps['dist_ben'] = cps['MEDICAID'] + cps['SSI'] + cps['SNAP'] + cps['VB']
cps['ratio'] = cps.dist_ben  * cps.s006 / (cps.dist_ben * cps.s006).sum() 
cps['other'] = cps.ratio * other_programs['Cost'].sum() / cps.s006

# Base calculator
recs = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
calc = Calculator(records=recs, policy=Policy(), verbose=False)
calc.advance_to_year(2014)
calc.calc_all()

# Calculator to measure lost revenue from SS repeal
r_ss = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
c_ss = Calculator(records=r_ss, policy=Policy(), verbose=False)
c_ss.records.e02400 = np.zeros(len(c_ss.records.e02400))
c_ss.advance_to_year(2014)
c_ss.calc_all()

# Lost Revenue
ss_lostrev = ((c_ss.records.combined - calc.records.combined) * c_ss.records.s006).sum()

cps_storage = copy.deepcopy(cps)

# Calculator with original tax refrom
recs_reform = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
pol_reform = Policy()
tax_reform = {
    2014: {
        '_ALD_StudentLoan_hc': [1.0],
        '_ALD_SelfEmploymentTax_hc': [1.0],
        '_ALD_SelfEmp_HealthIns_hc': [1.0],
        '_ALD_KEOGH_SEP_hc': [1.0],
        '_ALD_EarlyWithdraw_hc': [1.0],
        '_ALD_Alimony_hc': [1.0],
        '_ALD_Dependents_hc': [1.0],
        '_ALD_EducatorExpenses_hc': [1.0],
        '_ALD_HSADeduction_hc': [1.0],
        '_ALD_IRAContributions_hc': [1.0],
        '_ALD_DomesticProduction_hc': [1.0],
        '_ALD_Tuition_hc': [1.0],
        '_CR_RetirementSavings_hc': [1.0],
        '_CR_ForeignTax_hc': [1.0],
        '_CR_ResidentialEnergy_hc': [1.0],
        '_CR_GeneralBusiness_hc': [1.0],
        '_CR_MinimumTax_hc': [1.0],
        '_CR_AmOppRefundable_hc': [1.0],
        '_CR_AmOppNonRefundable_hc': [1.0],
        '_CR_SchR_hc': [1.0],
        '_CR_OtherCredits_hc': [1.0],
        '_CR_Education_hc': [1.0],
        '_II_em': [0.0],
        '_STD': [[0.0, 0.0, 0.0, 0.0, 0.0]],
        '_STD_Aged': [[0.0, 0.0, 0.0, 0.0, 0.0]],
        '_ID_Medical_hc': [1.0],
        '_ID_StateLocalTax_hc': [1.0],
        '_ID_RealEstate_hc': [1.0],
        '_ID_InterestPaid_hc': [1.0],
        '_ID_Casualty_hc': [1.0],
        '_ID_Miscellaneous_hc': [1.0],
        '_CDCC_c': [0.0],
        '_CTC_c': [0.0],
        '_EITC_c': [[0.0, 0.0, 0.0, 0.0]],
        '_LLC_Expense_c': [0.0],
        '_ETC_pe_Single': [0.0],
        '_ETC_pe_Married': [0.0]
    }
}
pol_reform.implement_reform(tax_reform)
calc_reform = Calculator(records=recs_reform, policy=pol_reform, verbose=False)
calc_reform.records.e02400 = np.zeros(len(calc_reform.records.e02400))
calc_reform.advance_to_year(2014)
calc_reform.calc_all()

# Revenue from tax reform
tax_rev = ((calc_reform.records.combined - calc.records.combined) * calc_reform.records.s006).sum()

# Total UBI Revenue
revenue = cps_rev + other_rev + ss_lostrev + tax_rev
revenue

# Number above and below 18
u18 = (calc_reform.records.nu18 * calc_reform.records.s006).sum()
abv18 = ((calc_reform.records.n1821 + calc_reform.records.n21) * calc_reform.records.s006).sum()

# Find original UBI amounts
ubi18, ubiu18 = ubi_amt(revenue, u18, abv18)
ubi18, ubiu18

# Find UBI after accounting for UBI tax revenue
diff = 9e99
ubi_tax_rev = 0
prev_ubi_tax_rev = 0
while abs(diff) >= 100:
    ubi18, ubiu18 = ubi_amt(revenue + ubi_tax_rev, u18, abv18)
    diff, ubi_tax_rev = ubi_finder(ubi18, ubiu18, 
                                   tax_reform=tax_reform, revenue=revenue,
                                   calc_reform=calc_reform)
    if diff > 0:
        ubi_tax_rev = prev_ubi_tax_rev * 0.5
    prev_ubi_tax_rev = ubi_tax_rev
ubi18, ubiu18

# Calculator with UBI and tax reform
recs_ubi1 = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
pol_ubi1 = Policy()
pol_ubi1.implement_reform(tax_reform)
ubi_ref = {
    2014: {
        '_UBI1': [ubiu18],
        '_UBI2': [ubi18],
        '_UBI3': [ubi18]
    }
}
pol_ubi1.implement_reform(ubi_ref)
calc_ubi1 = Calculator(records=recs_ubi1, policy=pol_ubi1, verbose=False)
calc_ubi1.records.e02400 = np.zeros(len(calc_ubi1.records.e02400))
calc_ubi1.advance_to_year(2014)
calc_ubi1.calc_all()

# Get MTR's
mtrs = calc_ubi1.mtr()

pd.options.display.float_format = '{:,.2f}'.format

table_data1 = prep_table_data(calc=calc_ubi1, calc_base=calc, mtrs=mtrs, bins='income')

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='all', bins='income')
table(table_data1, avg_ben, avg_ben_mult)

table_data2 = prep_table_data(calc=calc_ubi1, calc_base=calc, mtrs=mtrs, group='65 or over', bins='income')
#somehow s006 has to be the first variable in table? why?!!

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='65 or over', bins='income')
table(table_data2, avg_ben, avg_ben_mult)

table_data3 = prep_table_data(calc=calc_ubi1, calc_base=calc, mtrs=mtrs, group='under 65', bins='income')

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='under 65', bins='income')
table(table_data3, avg_ben, avg_ben_mult)

# Calculator with second reform policy
recs_reform2 = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
pol_reform2 = Policy()
pol_reform2.implement_reform(tax_reform)
tax_reform2 = {
    2014: {
        '_ALD_StudentLoan_hc': [1.0],
        '_ALD_SelfEmploymentTax_hc': [1.0],
        '_ALD_SelfEmp_HealthIns_hc': [1.0],
        '_ALD_KEOGH_SEP_hc': [1.0],
        '_ALD_EarlyWithdraw_hc': [1.0],
        '_ALD_Alimony_hc': [1.0],
        '_ALD_Dependents_hc': [1.0],
        '_ALD_EducatorExpenses_hc': [1.0],
        '_ALD_HSADeduction_hc': [1.0],
        '_ALD_IRAContributions_hc': [1.0],
        '_ALD_DomesticProduction_hc': [1.0],
        '_ALD_Tuition_hc': [1.0],
        '_CR_RetirementSavings_hc': [1.0],
        '_CR_ForeignTax_hc': [1.0],
        '_CR_ResidentialEnergy_hc': [1.0],
        '_CR_GeneralBusiness_hc': [1.0],
        '_CR_MinimumTax_hc': [1.0],
        '_CR_AmOppRefundable_hc': [1.0],
        '_CR_AmOppNonRefundable_hc': [1.0],
        '_CR_SchR_hc': [1.0],
        '_CR_OtherCredits_hc': [1.0],
        '_CR_Education_hc': [1.0],
        '_II_em': [0.0],
        '_STD': [[0.0, 0.0, 0.0, 0.0, 0.0]],
        '_STD_Aged': [[0.0, 0.0, 0.0, 0.0, 0.0]],
        '_ID_Medical_hc': [1.0],
        '_ID_StateLocalTax_hc': [1.0],
        '_ID_RealEstate_hc': [1.0],
        '_ID_InterestPaid_hc': [1.0],
        '_ID_Casualty_hc': [1.0],
        '_ID_Miscellaneous_hc': [1.0],
        '_CDCC_c': [0.0],
        '_CTC_c': [0.0],
        '_EITC_c': [[0.0, 0.0, 0.0, 0.0]],
        '_LLC_Expense_c': [0.0],
        '_ETC_pe_Single': [0.0],
        '_ETC_pe_Married': [0.0],
        '_II_rt2': [.10],
        '_II_rt3': [.10],
        '_II_rt4': [.10],
        '_II_rt5': [.10],
        '_II_rt6': [.10],
        '_II_rt7': [.50],
        '_II_brk1': [[50000, 100000, 50000, 50000, 100000]],
        '_II_brk2': [[50000, 100000, 50000, 50000, 100000]],
        '_II_brk3': [[50000, 100000, 50000, 50000, 100000]],
        '_II_brk4': [[50000, 100000, 50000, 50000, 100000]],
        '_II_brk5': [[50000, 100000, 50000, 50000, 100000]],
        '_II_brk6': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_rt2': [.10],
        '_PT_rt3': [.10],
        '_PT_rt4': [.10],
        '_PT_rt5': [.10],
        '_PT_rt6': [.10],
        '_PT_rt7': [.50],
        '_PT_brk1': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_brk2': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_brk3': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_brk4': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_brk5': [[50000, 100000, 50000, 50000, 100000]],
        '_PT_brk6': [[50000, 100000, 50000, 50000, 100000]],
        '_AMT_rt1': [0.0],
        '_AMT_rt2': [0.0]
    }
}
pol_reform2.implement_reform(tax_reform2)
calc_reform2 = Calculator(records=recs_reform2, policy=pol_reform2, verbose=False)
calc_reform2.records.e02400 = np.zeros(len(calc_reform2.records.e02400))
calc_reform2.advance_to_year(2014)
calc_reform2.calc_all()

# Revenue from tax reform
tax_rev2 = ((calc_reform2.records.combined - calc.records.combined) * calc_reform2.records.s006).sum()

revenue2 = cps_rev + other_rev + ss_lostrev + tax_rev2
revenue2

# Find original UBI amounts
ubi18, ubiu18 = ubi_amt(revenue2, u18, abv18)
ubi18, ubiu18

# Find UBI after accounting for UBI tax revenue
diff = 9e99
ubi_tax_rev = 0
prev_ubi_tax_rev = 0
while abs(diff) >= 300:
    ubi18, ubiu18 = ubi_amt(revenue2 + ubi_tax_rev, u18, abv18)
    diff, ubi_tax_rev = ubi_finder(ubi18, ubiu18, 
                                   tax_reform=tax_reform2, revenue=revenue2,
                                   calc_reform=calc_reform2)
    if diff > 0:
        ubi_tax_rev = prev_ubi_tax_rev * 0.5
    prev_ubi_tax_rev = ubi_tax_rev
    print diff
ubi18, ubiu18

# Calculator with UBI and tax reform
recs_ubi2 = Records('puf_benefits.csv', weights='puf_weights_new.csv', adjust_ratios='puf_ratios copy.csv')
pol_ubi2 = Policy()
pol_ubi2.implement_reform(tax_reform)
ubi_ref2 = {
    2014: {
        '_UBI1': [ubiu18],
        '_UBI2': [ubi18],
        '_UBI3': [ubi18]
    }
}
pol_ubi2.implement_reform(ubi_ref2)
calc_ubi2 = Calculator(records=recs_ubi2, policy=pol_ubi2, verbose=False)
calc_ubi2.records.e02400 = np.zeros(len(calc_ubi2.records.e02400))
calc_ubi2.advance_to_year(2014)
calc_ubi2.calc_all()

# Get MTR's
# try using baseline MTR
mtrs2 = calc.mtr()

table_data4 = prep_table_data(calc=calc_ubi2, calc_base=calc, mtrs=mtrs2, bins='income')

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='all', bins='income')
table(table_data4, avg_ben, avg_ben_mult)

table_data5 = prep_table_data(calc=calc_ubi2, calc_base=calc, mtrs=mtrs2, group='65 or over', bins='income')

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='65 or over', bins='income')
table(table_data5, avg_ben, avg_ben_mult)

table_data6 = prep_table_data(calc=calc_ubi2, calc_base=calc, mtrs=mtrs2, group='under 65', bins='income')

avg_ben, avg_ben_mult = cps_avg_ben(cps_storage, other_programs, group='under 65', bins='income')
table(table_data6, avg_ben, avg_ben_mult)

