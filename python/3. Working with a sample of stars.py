import q2
data = q2.Data('hyades_stars.csv', 'hyades_lines.csv')

sp = q2.specpars.SolvePars()
sp.grid = 'marcs'

q2.specpars.solve_all(data, sp, 'hyades_solution.csv', 'vestaOct')

sp.grid = 'odfnew'
sp.errors = True

pp = q2.specpars.PlotPars()
pp.afe = [-0.05, 0.35]
pp.figure_format = 'eps'
pp.directory = 'odfnew'

q2.specpars.solve_all(data, sp, 'hyades_solution_odfnew_err.csv',
                      'vestaOct', pp)

