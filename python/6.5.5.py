print(likelihood_ratio_test(ris_result,rs_result))

experimental_model = smf.mixedlm("P ~ Week+cond+Week*cond",data=PE_long,groups=PE_long['ID'], re_formula="~Week")
result = experimental_model.fit(method='nm', maxiter=200, full_output=True)
print(result.summary())

