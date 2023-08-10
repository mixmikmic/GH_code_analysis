
import pickle
import pandas as pd

# Handy list of the different types of encodings
encoding = ['latin1', 'iso8859-1', 'utf-8'][1]

# Change this to your data and saves folders
data_folder = r'../../data/'
saves_folder = r'../../saves/'

def load_object(obj_name):
    pickle_path = saves_folder + 'pickle/' + obj_name + '.pickle'
    try:
        object = pd.read_pickle(pickle_path)
    except:
        with open(pickle_path, 'rb') as handle:
            object = pickle.load(handle)
    
    return(object)


from sklearn.decomposition import PCA
import os

obj_path = saves_folder + 'pickle/gapminder_df.pickle'
if not os.path.isfile(obj_path):
    gapminder_df = pd.read_csv(data_folder + 'csv/gapminder.csv',
                               low_memory=False, encoding=encoding)
else:
    gapminder_df = load_object('gapminder_df')
gapminder_df.columns = ['country_name', 'income_per_person',
                        'alcohol_consumption', 'armed_forces_rate',
                        'breast_cancer_per_100th', 'co2_emissions',
                        'female_employment_rate', 'hiv_rate',
                        'internet_use_rate', 'life_expectancy',
                        'oil_per_person', 'polity_score',
                        'residential_electricity_per_person',
                        'suicide_per_100th', 'employment_rate',
                        'urban_rate']
number_column_list = list(set(gapminder_df.columns) - set(['country_name']))
elite_df = gapminder_df.dropna(how='any').copy()
elite_ndarray = PCA(n_components=2).fit_transform(elite_df[number_column_list])


# Classes, functions, and methods cannot be pickled
def store_objects(**kwargs):
    for obj_name in kwargs:
        if hasattr(kwargs[obj_name], '__call__'):
            raise RuntimeError('Functions cannot be pickled.')
        obj_path = saves_folder + 'pickle/' + str(obj_name)
        pickle_path = obj_path + '.pickle'
        if isinstance(kwargs[obj_name], pd.DataFrame):
            kwargs[obj_name].to_pickle(pickle_path)
        else:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(kwargs[obj_name], handle, pickle.HIGHEST_PROTOCOL)


obj_path = saves_folder + 'pickle/formal_name_dict.pickle'
if not os.path.isfile(obj_path):
    formal_name_dict = {}
    formal_name_dict['alcohol_consumption'] = '2008 alcohol consumption per adult (age 15+) in litres'
    formal_name_dict['armed_forces_rate'] = 'Armed forces personnel as a % of total labor force'
    formal_name_dict['breast_cancer_per_100th'] = '2002 breast cancer new cases per hundred thousand females'
    formal_name_dict['co2_emissions'] = '2006 cumulative CO2 emission in metric tons'
    formal_name_dict['employment_rate'] = '2007 total employees age 15+ as a % of population'
    formal_name_dict['female_employment_rate'] = '2007 female employees age 15+ as a % of population'
    formal_name_dict['hiv_rate'] = '2009 estimated HIV Prevalence % for Ages 15-49'
    formal_name_dict['income_per_person'] = '2010 Gross Domestic Product per capita in constant 2000 USD'
    formal_name_dict['internet_use_rate'] = '2010 Internet users per 100 people'
    formal_name_dict['life_expectancy'] = '2011 life expectancy at birth in years'
    formal_name_dict['oil_per_person'] = '2010 oil Consumption per capita in tonnes per year and person'
    formal_name_dict['polity_score'] = '2009 Democracy score as measured by Polity'
    formal_name_dict['residential_electricity_per_person'] = '2008 residential electricity consumption per person in kWh'
    formal_name_dict['suicide_per_100th'] = '2005 Suicide age adjusted per hundred thousand'
    formal_name_dict['urban_rate'] = '2008 urban population as a % of total'
    store_objects(formal_name_dict=formal_name_dict)
else:
    formal_name_dict = load_object('formal_name_dict')


obj_path = saves_folder + 'pickle/informal_name_dict.pickle'
if not os.path.isfile(obj_path):
    informal_name_dict = {}
    informal_name_dict['alcohol_consumption'] = 'alcohol consumption'
    informal_name_dict['armed_forces_rate'] = 'armed forces rate'
    informal_name_dict['breast_cancer_per_100th'] = 'breast cancer'
    informal_name_dict['co2_emissions'] = 'CO2 emissions'
    informal_name_dict['employment_rate'] = 'employment rate'
    informal_name_dict['female_employment_rate'] = 'female employment rate'
    informal_name_dict['hiv_rate'] = 'HIV rate'
    informal_name_dict['income_per_person'] = 'income per person'
    informal_name_dict['internet_use_rate'] = 'internet use rate'
    informal_name_dict['life_expectancy'] = 'life expectancy'
    informal_name_dict['oil_per_person'] = 'oil per person'
    informal_name_dict['polity_score'] = 'polity score'
    informal_name_dict['residential_electricity_per_person'] = 'residential electricity'
    informal_name_dict['suicide_per_100th'] = 'suicide rate'
    informal_name_dict['urban_rate'] = 'urban rate'
    store_objects(informal_name_dict=informal_name_dict)
else:
    informal_name_dict = load_object('informal_name_dict')


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Compute DBSCAN
db = DBSCAN(eps=7750000000, min_samples=2).fit(elite_ndarray)
labels = db.labels_

fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111, autoscale_on=True)
cmap = plt.get_cmap('viridis_r')
path_collection = ax.scatter(elite_ndarray[:, 0], elite_ndarray[:, 1],
                             s=elite_df['alcohol_consumption']*10,
                             c=elite_df['suicide_per_100th'],
                             edgecolors=(0, 0, 0), cmap=cmap)
kwargs = dict(textcoords='offset points', ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
for col, label, x, y in zip(labels, elite_df['country_name'],
                            elite_ndarray[:, 0], elite_ndarray[:, 1]):
    if (label == 'Japan'):
        annotation = plt.annotate(label, xy=(x, y), xytext=(20, 10), **kwargs)
    elif (label == 'United Kingdom'):
        annotation = plt.annotate(label, xy=(x, y), xytext=(20, 20), **kwargs)
    elif (label == 'China') or (label == 'United States'):
        annotation = plt.annotate(label, xy=(x, y), xytext=(-50, 20), **kwargs)
    elif (col == 1):
        annotation = plt.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
title_text = 'Dimension Reduced Scatterplot of the GapMinder '
title_text += 'Fields with Alcohol Consumption as the size and '
title_text += 'Suicide Rate as the Color'
text = plt.title(title_text)


def create_binned_categories(df, number_of_categories, column_name, prefix):
    
    # Get the percentiles
    out_categorical, percentiles_list = pd.cut([0, 1], number_of_categories, retbins=True)
    describe_series = df[column_name].describe(percentiles=percentiles_list[1:-1]).copy()

    # Get the bin list and group names
    bad_list = ['count', 'mean', 'std']
    if (number_of_categories % 2) == 1:
        bad_list += ['50%']
    index_list = [x for x in describe_series.index.tolist() if x not in bad_list]
    bin_list = describe_series.loc[index_list].tolist()

    # Create the extra column
    df[prefix+'_categories'] = pd.cut(df[column_name],
                                      bin_list).map(lambda x: (x.left + x.right)/2.)

    # Fix the bottom row
    null_series = df[prefix+'_categories'].isnull()
    df.loc[null_series, prefix+'_categories'] = df[~null_series][prefix+'_categories'].min()

    return df


from scipy.stats import pearsonr

class Statements(object):

    def __init__(self, df, qe_column, qr_column, md_column, **kwargs):
        prop_defaults = {
            'low_high': 'both',
            'explanation_list': ['Pearson’s correlation coefficient', '2-tailed p-value'],
            'verbose_HTML': '',
            'qe_formal_name': formal_name_dict[qe_column],
            'qr_formal_name': formal_name_dict[qr_column],
            'md_formal_name': formal_name_dict[md_column],
            'qe_informal_name': informal_name_dict[qe_column],
            'qr_informal_name': informal_name_dict[qr_column],
            'md_informal_name': informal_name_dict[md_column],
        }

        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        self.df = df
        self.qe_column = qe_column
        self.qr_column = qr_column
        self.md_column = md_column
        self.pearsonr_tuple = pearsonr(df[qe_column], df[qr_column])
        self.pearson_r = self.pearsonr_tuple[0]
        self.coefficient_of_determination = self.pearson_r**2
        self.percent_predictable = self.coefficient_of_determination*100
        self.p_value = self.pearsonr_tuple[1]
        if self.low_high == 'both':
            self.moderator_statement = ('combined categories ' +
                                   ' and '.join(self.df['md_categories'].unique().map(lambda x: str(x))))
        else:
            self.moderator_statement = 'category ' + str(self.df['md_categories'].unique().tolist()[0])
        if self.pearson_r > 0:
            self.adjective_positive = 'positive'
        else:
            self.adjective_positive = 'negative'
        self.pearsonr_statement = str('%.2f' % self.pearson_r)
        if self.coefficient_of_determination >= 0.25:
            self.adverb_strong = 'strongly'
        else:
            self.adverb_strong = 'weakly'
        self.cod_statement = str('%.2f' % self.coefficient_of_determination)
        self.percent_statement = str('%.1f' % self.percent_predictable)
        if self.p_value < 0.0001:
            self.pvalue_statement = '<0.0001'
        else:
            self.pvalue_statement = '=' + str('%.4f' % self.p_value)
        if self.p_value < 0.05:
            self.adverb_significant = 'significantly'
        else:
            self.adverb_significant = 'insignificantly'


def round_down(num, divisor):
    
    return num - (num%divisor)

def round_up(num, divisor):
    
    return num - (num%divisor) + divisor


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

def add_joint_plot(sts):

    # Turn interactive plotting off
    plt.ioff()

    print()
    scatter_kws = dict(s=0, edgecolors='w')
    joint_kws = dict(scatter_kws=scatter_kws)
    xlim_multiple = round_up((sts.df[sts.qe_column].max() - sts.df[sts.qe_column].min()) / 5, 10)
    xlim = (round_down(sts.df[sts.qe_column].min(), xlim_multiple), round_up(sts.df[sts.qe_column].max(), xlim_multiple))
    ylim_multiple = round_up((sts.df[sts.qr_column].max() - sts.df[sts.qr_column].min()) / 5, 10)
    ylim = (round_down(sts.df[sts.qr_column].min(), ylim_multiple), round_up(sts.df[sts.qr_column].max(), ylim_multiple))
    joint_grid = sns.jointplot(x=sts.qe_column, y=sts.qr_column, data=sts.df, size=13, space=0,
                               stat_func=None, kind='reg',
                               joint_kws=joint_kws, marginal_kws=dict(bins=15, rug=True))
    joint_grid.ax_joint.set_autoscale_on(b=True)
    joint_grid.ax_marg_x.set_autoscale_on(b=True)
    joint_grid.ax_marg_y.set_autoscale_on(b=True)
    
    ax_joint_xlim = joint_grid.ax_joint.get_xlim()
    ax_joint_ylim = joint_grid.ax_joint.get_ylim()
    
    ax_marg_x_xlim = joint_grid.ax_marg_x.get_xlim()
    ax_marg_x_ylim = joint_grid.ax_marg_x.get_ylim()
    
    ax_marg_y_xlim = joint_grid.ax_marg_y.get_xlim()
    ax_marg_y_ylim = joint_grid.ax_marg_y.get_ylim()

    # Set the axes and title text
    xlabel = sts.qe_formal_name + ' (Explanatory Variable)'
    ylabel = sts.qr_formal_name + ' (Response Variable)'
    joint_grid.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
    plot_title_text = ('Scatterplot for the association between ' +
                       sts.qe_informal_name + ' and ' +
                       sts.qr_informal_name + ', colored and sized with ' +
                       sts.md_informal_name)
    joint_grid.fig.suptitle(plot_title_text)

    # Re-color and resize the plot points
    cmap = plt.get_cmap('viridis_r')
    scatter_kws = dict(s=sts.df[sts.md_column]*5, edgecolors=(0, 0, 0), cmap=cmap,
                       color=None, c=sts.df[sts.md_column])
    joint_grid = joint_grid.plot_joint(sns.regplot, fit_reg=False, scatter_kws=scatter_kws)
    joint_grid.ax_joint.set_autoscale_on(b=True)
    joint_grid.ax_marg_x.set_autoscale_on(b=True)
    joint_grid.ax_marg_y.set_autoscale_on(b=True)
    
    joint_grid.ax_joint.set_xlim(ax_joint_xlim)
    joint_grid.ax_marg_x.set_xlim(ax_marg_x_xlim)
    joint_grid.ax_marg_y.set_xlim(ax_marg_y_xlim)
    
    joint_grid.ax_joint.set_ylim(ax_joint_ylim)
    joint_grid.ax_marg_x.set_ylim(ax_marg_x_ylim)
    joint_grid.ax_marg_y.set_ylim(ax_marg_y_ylim)

    # Set the annotations
    r_squared = lambda a, b: pearsonr(a, b)[0] ** 2
    joint_grid = joint_grid.annotate(r_squared, template='{stat}: {val:.2f}', stat='$R^2$',
                                     loc='upper left', fontsize=18, bbox_to_anchor=(-0.07, 1.0),
                                     frameon=False)
    joint_grid.ax_joint.set_autoscale_on(b=True)
    joint_grid.ax_marg_x.set_autoscale_on(b=True)
    joint_grid.ax_marg_y.set_autoscale_on(b=True)
    
    joint_grid.ax_joint.set_xlim(ax_joint_xlim)
    joint_grid.ax_marg_x.set_xlim(ax_marg_x_xlim)
    joint_grid.ax_marg_y.set_xlim(ax_marg_y_xlim)
    
    joint_grid.ax_joint.set_ylim(ax_joint_ylim)
    joint_grid.ax_marg_x.set_ylim(ax_marg_x_ylim)
    joint_grid.ax_marg_y.set_ylim(ax_marg_y_ylim)
    
    kwargs = dict(textcoords='offset points', ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    for label, x, y in zip(sts.df['country_name'], sts.df[sts.qe_column], sts.df[sts.qr_column]):
        if (label == 'United States'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-50, -35), **kwargs)
        elif (label == 'Argentina'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
        elif (label == 'Pakistan'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
        elif (label == 'Bangladesh'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(25, -25), **kwargs)
            print('Bangladesh', x, y)
        elif (label == 'Thailand'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(25, -25), **kwargs)
        elif (label == 'China'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(50, -20), **kwargs)
        elif (label == 'Qatar'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(20, 10), **kwargs)
        elif (label == 'Korea, Rep.'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(20, 10), **kwargs)
        elif (label == 'Slovak Repulic'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
        elif (label == 'Norway'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
            print('Norway', x, y)
        elif (label == 'Denmark'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)
        elif (label == 'New Zealand'):
            annotation = joint_grid.ax_joint.annotate(label, xy=(x, y), xytext=(-10, 20), **kwargs)

    # Save the new figure then close it so it never gets displayed
    file_name = ('../../saves/png/plot_' + sts.qe_column + '_' + sts.qr_column +
                 '_' + sts.md_column + '_' + sts.low_high + '.png')
    joint_grid.savefig(file_name)
    plt.close(joint_grid.fig)

    sts.verbose_HTML += '<p><image src="' + file_name + '" /></p>'

    # Display all "open" (non-closed) figures
    #plt.show()

    return sts.verbose_HTML


from IPython.display import HTML

sts = Statements(create_binned_categories(elite_df, 2, 'female_employment_rate','md'),
                 'residential_electricity_per_person', 'breast_cancer_per_100th', 'female_employment_rate')
display(HTML(add_joint_plot(sts)))


from IPython.display import HTML

sts = Statements(create_binned_categories(elite_df, 2, 'female_employment_rate','md'),
                 'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate')
display(HTML(add_joint_plot(sts)))


from scipy.stats import norm

# Pearson’s Correlation Coefficient
def model_interpretation(sts, sample_name='the population', verbose=False):
        
    if verbose:
        sts.verbose_HTML += ('<p>Association between ' + 
                         sts.qe_informal_name + ' and ' + 
                         sts.qr_informal_name + ', with ' + 
                         sts.md_informal_name + ' moderator ' +
                         sts.moderator_statement + '<ul>')
        for x, y in zip(sts.explanation_list, sts.pearsonr_tuple):
            sts.verbose_HTML += '<li>{}: {}</li>'.format(x, y)
        sts.verbose_HTML += '</ul></p>'
        
    if sts.low_high == 'both':
        pearsonr_HTML = ('<h3>Model Interpretation for Pearson’s Correlation Coefficient Tests, Testing a Potential Moderator:</h3>' +
                         '<p>The Pearson’s Correlation Coefficient revealed that among ')
    else:
        pearsonr_HTML = ('<p>The Pearson’s Correlation Coefficient exclusive to the ' +
                         sts.moderator_statement + ' of the ' +
                         sts.md_informal_name + ' moderator revealed that among ')
    pearsonr_HTML += (sample_name + ', ' +
                      sts.qe_formal_name + ' (quantitative explanatory variable), and ' +
                      sts.qr_formal_name + ' (quantitative response variable) were ' +
                      sts.adverb_significant + ' associated, in a ' +
                      sts.adverb_strong + ' ' +
                      sts.adjective_positive + r' manner, $r=' +
                      sts.pearsonr_statement + ', p' +
                      sts.pvalue_statement + r'$. This means that if we know the ' +
                      sts.qe_informal_name + ' of ' +
                      sts.moderator_statement + ', we can predict ' +
                      sts.percent_statement + '% of the ' +
                      sts.qr_informal_name + '.</p>')
    
    if verbose:
        sts.verbose_HTML = add_joint_plot(sts)
    
    return HTML(pearsonr_HTML + sts.verbose_HTML)


sample_name = 'the sample of ' + str(elite_df.shape[0]) + ' countries from GapMinder.org'
sts = Statements(elite_df,
                 'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate')
model_interpretation(sts, sample_name=sample_name, verbose=True)


# A response variable corresponds to a dependent variable while
# an explanatory variable corresponds to an independent variable
def moderator_conclusion(both_sts, low_sts, high_sts):
    if high_sts.coefficient_of_determination > both_sts.coefficient_of_determination:
        verb_high_increased = 'increased'
    elif high_sts.coefficient_of_determination < both_sts.coefficient_of_determination:
        verb_high_increased = 'decreased'
    else:
        verb_high_increased = 'stayed the same'
    if low_sts.coefficient_of_determination > both_sts.coefficient_of_determination:
        verb_low_increased = 'increased'
    elif low_sts.coefficient_of_determination < both_sts.coefficient_of_determination:
        verb_low_increased = 'decreased'
    else:
        verb_low_increased = 'stayed the same'
    summary_HTML = ('<h3>Summary</h3><p>The effect of the moderating (' +
                    both_sts.md_informal_name + ') variable is characterized ' +
                    'statistically as an interaction; that is, a categorical (in this case low or high) ' +
                    'variable that affects the strength of the relation between dependent (' +
                    both_sts.qr_informal_name + ') and independent (' +
                    both_sts.qe_informal_name + ') variables. In our study, the strength of the relation ' +
                    '(coefficient of determination) between ' +
                    both_sts.qr_informal_name + ' and ' +
                    both_sts.qe_informal_name + ', when considering only the high ' +
                    high_sts.moderator_statement + ' of ' +
                    high_sts.md_informal_name + ', ' +
                    verb_high_increased + ' from ' +
                    both_sts.cod_statement + ' to ' +
                    high_sts.cod_statement + '. Similarly, when considering only the low ' +
                    low_sts.moderator_statement + ' of ' +
                    low_sts.md_informal_name + ', ' +
                    verb_low_increased + ' from ' +
                    both_sts.cod_statement + ' to ' +
                    low_sts.cod_statement + '.')
    
    return HTML(summary_HTML)


both_sts = Statements(create_binned_categories(elite_df, 2, 'female_employment_rate','md'),
                      'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate')
low_sts = Statements(both_sts.df[(both_sts.df['md_categories'] == both_sts.df['md_categories'].min())].copy(),
                     'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate',
                     low_high='low')
high_sts = Statements(both_sts.df[(both_sts.df['md_categories'] == both_sts.df['md_categories'].max())].copy(),
                     'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate',
                      low_high='high')
display(moderator_conclusion(both_sts, low_sts, high_sts))


def pearsons_with_moderator(sts, source_name='GapMinder.org', verbose=False):
    
    
    both_df = create_binned_categories(sts.df, 2, sts.md_column,'md')
    both_sts = Statements(both_df, sts.qe_column, sts.qr_column, sts.md_column)
    sample_name = 'the sample of ' + str(both_df.shape[0]) + ' countries from ' + source_name
    display(model_interpretation(both_sts, sample_name, verbose))
    
    match_series = (both_df['md_categories'] == both_df['md_categories'].min())
    low_df = both_df[match_series].copy()
    low_sts = Statements(low_df, sts.qe_column, sts.qr_column, sts.md_column, low_high='low')
    sample_name = 'the sample of ' + str(low_df.shape[0]) + ' low category countries from ' + source_name
    display(model_interpretation(low_sts, sample_name, verbose))
    
    match_series = (both_df['md_categories'] == both_df['md_categories'].max())
    high_df = both_df[match_series].copy()
    high_sts = Statements(high_df, sts.qe_column, sts.qr_column, sts.md_column, low_high='high')
    sample_name = 'the sample of ' + str(high_df.shape[0]) + ' high category countries from ' + source_name
    display(model_interpretation(high_sts, sample_name, verbose))
    
    display(moderator_conclusion(both_sts, low_sts, high_sts))


sts = Statements(elite_df,
                 'internet_use_rate', 'breast_cancer_per_100th', 'female_employment_rate')
pearsons_with_moderator(sts, verbose=True)


sts = Statements(elite_df,
                 'residential_electricity_per_person', 'breast_cancer_per_100th', 'female_employment_rate')
pearsons_with_moderator(sts, verbose=True)


sample_name = 'the sample of ' + str(elite_df.shape[0]) + ' countries from GapMinder.org'
rows_list = []
column_set_list = []
for qe_column in number_column_list:
    for qr_column in number_column_list:
        if qe_column is not qr_column:
            column_set = set([qe_column, qr_column])
            if column_set not in column_set_list:
                column_set_list.append(column_set)
                row_dict = {}
                pearsonr_tuple = pearsonr(elite_df[qe_column], elite_df[qr_column])
                pearson_r = pearsonr_tuple[0]
                coefficient_of_determination = pearson_r**2
                row_dict['qe_column'] = qe_column
                row_dict['qr_column'] = qr_column
                row_dict['coefficient_of_determination'] = coefficient_of_determination
                rows_list.append(row_dict)
determinations_df = pd.DataFrame(rows_list, columns=['qe_column', 'qr_column', 'coefficient_of_determination'])
determinations_df.sort_values('coefficient_of_determination', ascending=False).head(10)



