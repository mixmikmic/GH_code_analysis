import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().magic('matplotlib inline')

# latent %, sensible %, ocean %, aerosol_effect, nh_precip_suppression

# CanESM2  strong, clear

data_points = {'CCSM4': (35, 25, 40, 'weak', 'none'),
               'CSIRO-Mk3-6-0': (60, 20, 20, 'moderate', 'weak'),
               'FGOALS-g2': (60, 30, 10, 'weak', 'clear'),
               'GFDL-CM3': (50, 25, 25, 'strong', 'clear'),
               'GFDL-ESM2M': (70, 20, 10, 'weak', 'none'),
               'GISS-E2-H, p1': (33.3, 33.3, 33.3, 'weak', 'clear'),
               'GISS-E2-H, p3': (33, 33, 33, 'weak', 'clear'),
               'GISS-E2-R, p1': (40, 40, 20, 'weak', 'weak'), 
               'GISS-E2-R, p3': (50, 25, 25, 'weak', 'weak'),
               'IPSL-CM5A-LR': (50, 45, 5, 'moderate', 'clear'),
               'NorESM1-M': (20, 35, 45, 'moderate', 'clear')
              }

aerosol_alpha = {'weak': 0.2, 'moderate': 0.6, 'strong': 1.0}
precip_alpha = {'none': 0.2, 'weak': 0.6, 'clear': 1.0}

def create_figure(custom=None):
    """Create the 3D plot"""

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111, projection='3d')

    for model_name, data in data_points.items():
        x, y, z, aerosol_effect, nh_precip_suppression = data
        
        if custom == 'aerosol_effect':
            alpha = aerosol_alpha[aerosol_effect]
        elif custom == 'nh_precip_suppression':
            alpha = precip_alpha[nh_precip_suppression]
        else:
            alpha = 1.0
        
        ax.plot([x], [y], [z], label=model_name, marker='o', alpha=alpha, ms=10, linestyle='None')

    ax.set_xlabel('latent heat flux (%)')
    ax.set_ylabel('sensible heat flux (%)')
    ax.set_zlabel('heat flux into ocean (%)')

    ax.legend(loc=3)
    plt.title('Estimated compensation for skewed net radiative flux')
    plt.show()

create_figure()

create_figure(custom='aerosol_effect')

create_figure(custom='nh_precip_suppression')



