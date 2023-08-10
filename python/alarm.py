from cpt import * 

V_burglar = Variable('burglar', [True, False])
V_storm = Variable('storm', [True, False])
V_report = Variable('report', [True, False])
V_window = Variable('window', [True, False])
V_alarm = Variable('alarm', [True, False])

P_burglar = CPT([0.001, 0.999], [V_burglar])
P_storm = CPT([0.01, 0.99], [V_storm])

# P(report | storm)
CP_report = CPT([0.9, 0.02, 0.1, 0.98], [V_report, V_storm])

# P(window | storm, burglar)
CP_window = CPT([0.95, 0.9, 0.5, 0.0, 0.05, 0.1, 0.5, 1.0],
                [V_window, V_burglar, V_storm])

# P(alarm | window)
CP_alarm = CPT([0.95, 0.01, 0.05, 0.99], [V_alarm, V_window])

# Print some tables
print(P_burglar)

# Join: Compute P(report|storm), P(storm) => P(storm, report)
P_SR = join(CP_report, P_storm)
print(P_SR)

# Marginalize: Compute P(report) from P(storm, report)
P_R = marginalize(P_SR, ['report'])
print(P_R)

# Compute the joint distribution over all variables
P_DSRAW = reduce(join, [P_burglar, P_storm, CP_report, CP_window, CP_alarm])
print(P_DSRAW)

# Add observations alarm=True and report=False
P_posteriori = eliminate(eliminate(P_DSRAW, 'alarm', True), 'report', False)
# Marginalize to burglar
P_burglar_post = marginalize(P_posteriori, 'burglar')
print(P_burglar_post)

f = FactorGraph([V_burglar, V_storm, V_window, V_alarm, V_report], 
                [P_burglar, P_storm, CP_window, CP_alarm, CP_report])

f.forward_backward_pass()

f.posterior(V_alarm)

NEW_P_alarm = CPT([1.0, 0.0], [V_alarm])
NEW_P_report = CPT([0.0, 1.0], [V_report])

f2 = FactorGraph([V_burglar, V_storm, V_window, V_alarm, V_report], 
                 [P_burglar, P_storm, CP_window, CP_alarm, CP_report,
                  NEW_P_alarm, NEW_P_report])

f2.forward_backward_pass()
print(f2.posterior(V_burglar))

