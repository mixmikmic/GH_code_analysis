import pandas as pd
import dateutil.parser
#%matplotlib notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib

# Set ipython's max row display
pd.set_option('display.max_row', 10)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# Set mmaplotlib style to something nice
matplotlib.style.use('ggplot')

def splitter (package):
    """Split a Debian package name in its components
    
    :param package: full Debian package name
    :return:        pd.Series (name, version, epoch, upstream, revision)
    
    """

    name = package.split(':',1)[0]
    version = package.split(':',1)[1]
    if ':' in version:
        (epoch, rest) = version.split(':',1)
    else:
        (epoch, rest) = ('', version)
    if '-' in rest:
        (upstream, revision) = rest.rsplit('-',1)
    else:
        (upstream, revision) = (rest, '')
    return pd.Series({'name': name, 'version': version, 'epoch': epoch,
                     'upstream': upstream, 'revision': revision})

# Columnos of interst in the CSV to read
parameters = ['different_lines', 'common_lines', 'different_files', 'common_files',
              'diff_commits', 'normal_effort']
# Read results (all lines start with "CSV")
df = pd.read_csv("results/results.csv")

# Convert dates to datetime
df['datetime'] = df['date'].apply(dateutil.parser.parse)
# Add fields for the components of the Debian package name
# For each package we will have now
# ['package', 'name', 'version', 'epoch', 'upstream', 'revision']
df = df.merge(df['package'].apply(splitter), left_index=True, right_index=True)

# Names of packages in dataframe
pkg_names = df['name'].unique()
# Number of versions (all packages)
pkg_count = len(df.index)
# Number of versions (for each package)
pkg_name_counts = df['name'].value_counts()

print('Analyzing a total of {} package versions'.format(pkg_no))
print('Package names:', ', '.join(pkg_names))
print('Parameters available:', ', '.join(list(df)))
print('Parameters to analyze:', ', '.join(parameters))
print('Versions for each package:')
for (name, count) in pkg_name_counts.items():
    print('    {}: {}'.format(name, count))

pkgs = {}
for name in pkg_names:
#for name in ['acl']:
    pkgs[name] = df[df['name'] == name]
    pkgs[name] = pkgs[name].sort_values(by=['epoch', 'upstream', 'revision'])

for name in pkgs:
    df_plot = pkgs[name][['datetime'] + parameters]
    plt.figure()
    ax = df_plot.plot(x='datetime', subplots=True, grid=True, layout=(4,2), sharex=True,
                      kind='line', title='Package: '+name, figsize=(11,6))

def create_subplots(parameters):
    """Create subplots for each parameter
    
    :param parameters: list of parameters
    :return:           dictionary, with parameters as keys, subplots as values
    
    """
    
    (fig, axes) = plt.subplots((len(parameters)+1)//2, 2, figsize=(11, 8));
    current_ax = [0,0]
    params_ax = {}
    for parameter in parameters:
        params_ax[parameter] = axes[current_ax[0]][current_ax[1]]
        if current_ax[1] == 0:
            current_ax[1] = 1
        else:
            current_ax[0] += 1
            current_ax[1] = 0
    return params_ax

for name in pkgs:
#for name in ['acl']:
    df_pkg = pkgs[name]
    df_pkg = df_pkg.sort_values(by=['epoch', 'upstream', 'revision'])
    df_pkg_metrics = df_pkg[['datetime', 'upstream'] + parameters]
    upstreams = df_pkg['upstream'].unique()
    #print(upstreams)
    params_ax = create_subplots(parameters)
    for upstream in upstreams:
        df_plot = df_pkg_metrics[df_pkg_metrics['upstream'] == upstream]
        for parameter in parameters:
            df_plot.plot(x='datetime', y=parameter, legend=False, sharex=True,
                         kind='line', title='Package: {} ({})'.format(name, parameter),
                         ax=params_ax[parameter], marker='o')



