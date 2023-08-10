from matminer.datasets.dataframe_loader import load_elastic_tensor, load_dielectric_constant
from pymatgen import Composition
from matminer.figrecipes.plot import PlotlyFig
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
import pandas as pd
import pprint

# Get data
cdr = CitrineDataRetrieval() # Note that your Citrine API key must be set
cols = ['chemicalFormula', 'Electrical resistivity', 'Seebeck coefficient',
            'Thermal conductivity', 'Thermoelectric figure of merit (zT)']
df_te = cdr.get_dataframe(criteria={'data_type': 'experimental',
                                    'data_set_id': 150557},
                          properties=['Seebeck coefficient'],
                          secondary_fields=True)
df_te = df_te[cols].apply(pd.to_numeric, errors='ignore')
# Filter data based on resistivities between 0.0005 and 0.1 and
# Seebeck coefficients less than 500
df_te = df_te[(df_te['Electrical resistivity'] > 5e-4) &               (df_te['Electrical resistivity'] < 0.1)]
df_te = df_te[abs(df_te['Seebeck coefficient']) < 500]
df_te = df_te.rename(columns={'Thermoelectric figure of merit (zT)': 'zT'})

# Generate plots
pf = PlotlyFig(df_te, x_scale='log', fontfamily='Times New Roman',
               hovercolor='white', x_title='Electrical Resistivity (cm/S)',
               y_title='Seebeck Coefficient (uV/K)',
               colorbar_title='Thermal Conductivity (W/m.K)',
               mode='notebook')

pf.xy(('Electrical resistivity', 'Seebeck coefficient'),
      labels='chemicalFormula',
      sizes='zT',
      colors='Thermal conductivity',
      color_range=[0, 5])

df = load_elastic_tensor()

pf = PlotlyFig(df, title='Elastic data', mode='offline', 
               x_scale='log', y_scale='log')

# Lets plot offline (the default) first. An html file will be created.
pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

pf.set_arguments(show_offline_plot=False, filename="myplot.html")
pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

# pf.set_arguments(mode='static', api_key=YOUR_API_KEY,
#                 username=YOUR_USERNAME,
#                 filename="my_PlotlyFig_plot.jpeg")
# pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

# pf.set_arguments(mode='online')
# pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

pf.set_arguments(mode='notebook')
pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

fig = pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula',
            return_plot=True)
print("Here's our returned figure!")
# pprint.pprint(fig)

# Edit the figure and plot it with the current plot mode (online):
fig['layout']['hoverlabel']['bgcolor'] = 'pink'
fig['layout']['title'] = 'My Custom Elastic Data Figure'
pf.set_arguments(mode='notebook')
pf.create_plot(fig)

pf = PlotlyFig(df=df,
               # api_key=api_key,
               # username=username,
               mode='notebook',
               title='Comparison of Bulk Modulus and Shear Modulus',
               x_title='Shear modulus (GPa)',
               y_title='Bulk modulus (GPa)',
               colorbar_title='Poisson Ratio',
               fontfamily='Raleway',
               fontscale=0.75,
               fontcolor='#283747',
               ticksize=30,
               colorscale="Reds",
               hovercolor='white',
               hoverinfo='text',
               bgcolor='#F4F6F6',
               margins=110,
               pad=10)

pf.xy(('G_VRH', 'K_VRH'), labels='material_id', colors='poisson_ratio')

# We can also use LaTeX if we use Plotly online/static
# pf.set_arguments(title="$\\text{Origin of Poisson Ratio } \\nu $",
#                  y_title='$K_{VRH} \\text{(GPa)}$',
#                  x_title='$G_{VRH} \\text{(GPa)}$',
#                  colorbar_title='$\\nu$',
#                  api_key=YOUR_API_KEY, username=YOUR_USERNAME)
# pf.xy(('G_VRH', 'K_VRH'), labels='material_id', colors='poisson_ratio')

