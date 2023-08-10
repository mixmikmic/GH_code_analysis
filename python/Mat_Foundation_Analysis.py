import pandas as pd
reactions = pd.read_excel('MatFoundationReactions-1500mm.xlsx')
reactions.describe()

from SAPModule.Graphics import plot_contour_map
from SAPModule.Design import design_steel_rein
get_ipython().run_line_magic('matplotlib', 'inline')

plot_contour_map(file_name='MatFoundationReactions-1500mm.xlsx',
                stress_name = 'Dead_(kN)',
                level = [0, 220, 30],
                color = 'jet',
                title = 'Ultimate Bearing Capacity of Mat Foundation Heat Map',
                fsize = [15,15],
                contour_level=30)

shell_forces = pd.read_excel('ShellForcesCLEAN-1500mm.xlsx')
shell_forces.describe()

plot_contour_map(file_name='ShellForcesCLEAN-1500mm.xlsx',
                stress_name = 'M11',
                level = [-100, 5000, 30],
                color = 'jet',
                title = 'Factored Moment (kN.m / m) Along X-axis Factored Dead + Live Stresses on Mat Foundation Heat Map',
                fsize = [15,15],
                contour_level=15)

Moment = 1.62*1.39*5000
design_steel_rein(Mu = Moment, b = 3*1620, d = 1500, db = 28, layers = 1, cover = 75)

Moment = 1.62*1.39*3000
design_steel_rein(Mu = Moment, b = 5200, d = 1500, db = 28, layers = 1, cover = 75)

Moment = 1.62*1.39*800
design_steel_rein(Mu = Moment, b = 2*1620, d = 1500, db = 28, layers = 1, cover = 75)

plot_contour_map(file_name='ShellForcesCLEAN-1500mm.xlsx',
                stress_name = 'M22',
                level = [-650, 4200, 30],
                color = 'jet',
                title = 'Factored Moment (kN.m / m) Along Y-axis Factored Dead + Live Stresses on Mat Foundation Heat Map',
                fsize = [15,15])

Moment = 1.32*1.62*3500
design_steel_rein(Mu = Moment, b = 4*1390, d = 1500, db = 28, layers = 1, cover = 75)

Moment = 3*1.32*1.62*2500
design_steel_rein(Mu = Moment, b = 4*1390, d = 1500, db = 28, layers = 1, cover = 75)

Moment = 1.32*3*1.62*3000
design_steel_rein(Mu = Moment, b = 4*1390, d = 1500, db = 28, layers = 1, cover = 75)

Moment = 1.32*3*600*1.62
design_steel_rein(Mu = Moment, b = 4*1390, d = 1500, db = 28, layers = 1, cover = 75)

plot_contour_map(file_name='ShellForcesCLEAN-1500mm.xlsx',
                stress_name = 'VMax',
                level = [5, 4400, 30],
                color = 'jet',
                title = 'Maximum Punching Shear Force (kN / m) on Mat Foundation Heat Map',
                fsize = [15,15])

