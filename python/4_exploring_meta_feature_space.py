# Load required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2

# Set figure display options
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(context='notebook', style='darkgrid')
sns.set(font_scale=1.3)

# Set database credentials
db_name1 = 'section1_db'
usernm = 'redwan'
host = 'localhost'
port = '5432'
#pwd = ''

# Connect to a database
con1 = psycopg2.connect(
    database=db_name1, 
    host='localhost',
    user=usernm,
    password=pwd
)

# Query all data from the campaign section containing "About this projects"
sql_query1 = 'SELECT * FROM section1_db;'
section1_df_full = pd.read_sql_query(sql_query1, con1)

# Select the meta features
features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
            'num_exclms', 'percent_exclms', 'num_apple_words',
            'percent_apple_words', 'avg_words_per_sent', 'num_paragraphs',
            'avg_sents_per_paragraph', 'avg_words_per_paragraph',
            'num_images', 'num_videos', 'num_youtubes', 'num_gifs',
            'num_hyperlinks', 'num_bolded', 'percent_bolded']

# Separate the funded and unfunded projects
funded_projects = section1_df_full[section1_df_full['funded'] == True]
failed_projects = section1_df_full[section1_df_full['funded'] == False]

# Compute pairwise correlation coefficients between each meta feature
funded_corr = funded_projects[features].corr()
failed_corr = failed_projects[features].corr()

# Set the diagonal values to zero for both correlation coefficient tables
for i in range(len(funded_corr)):
    funded_corr.iloc[i, i] = 0
    failed_corr.iloc[i, i] = 0

# Plot a heatmap of the correlation coefficients of the meta features within 
# funded projects
plt.figure(figsize = (8, 5))
sns.heatmap(funded_corr);

# Plot a heatmap of the correlation coefficients of the meta features within 
# unfunded projects
plt.figure(figsize = (8, 5))
sns.heatmap(failed_corr);

# Plot a scatterplot of the 'avg_sents_per_paragraph' vs. 
# 'avg_words_per_paragraph'
sns.lmplot(
    data=section1_df_full,
    x='avg_sents_per_paragraph',
    y='avg_words_per_paragraph',
    hue='funded',
    fit_reg=False,
    palette='Set1'
).set(xlabel='average sentences/paragraph', ylabel='average words/paragraph');

# Plot a heatmap of the correlation coefficients of the meta features between
# funded and unfunded projects
plt.figure(figsize = (8, 5))
sns.heatmap(funded_corr - failed_corr);

# Rename the 'funded' column to ' ' as to hide the legend title
renamed_df = section1_df_full.rename(columns={'funded': ' '})

# Rename the classes
renamed_df.loc[renamed_df[' '] == True, ' '] = 'Funded'
renamed_df.loc[renamed_df[' '] == False, ' '] = 'Not funded'

# Make the font scale larger
sns.set(font_scale=2.2)

# Setup a FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1', aspect=1.4)

# Plot a kde plot for # of sentences
fig.map(sns.kdeplot, 'num_sents', shade=True)     .set(
        xlim=(-20, 300),
        xlabel='# of sentences',
        yticks=[],
        ylabel='relative frequency'
    )

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1', aspect=1.4)

# Plot a kde plot for # of words
fig.map(sns.kdeplot, 'num_words', shade=True)     .set(
        xlim=(-250, 4000),
        xlabel='# of words',
        yticks=[],
        ylabel='relative frequency'
    );

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of all-caps words
fig.map(sns.kdeplot, 'num_all_caps', shade=True).add_legend();

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for % of all-caps words
fig.map(sns.kdeplot, 'percent_all_caps', shade=True)     .add_legend()     .set(ylim=(0, 20));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1', aspect=1.4)

# Plot a kde plot for # of exclamation marks
fig.map(sns.kdeplot, 'num_exclms', shade=True)     .set(
        xlim=(-5, 70),
        xlabel='# of exclamation marks',
        yticks=[],
        ylabel='relative frequency'
    );

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for % of exclamation marks
fig.map(sns.kdeplot, 'percent_exclms', shade=True)     .add_legend()     .set(ylim=(0, 45));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1', aspect=1.615)

# Plot a kde plot for # of innovation words
fig.map(sns.distplot, 'num_apple_words', kde=False)     .set(
        xlim=(-1, 10),
        xlabel='# of innovation words',
        yticks=[],
        ylabel='relative frequency'
    );

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for % of innovation words
fig.map(sns.kdeplot, 'percent_apple_words', shade=True)     .add_legend()     .set(ylim=(0, 1200));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot of average # of words per sentence
fig.map(sns.kdeplot, 'avg_words_per_sent', shade=True)     .add_legend()     .set(ylim=(0, 0.1));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of paragraphs
fig.map(sns.kdeplot, 'num_paragraphs', shade=True)     .add_legend()     .set(xlim=(-10, 150), xlabel='# of paragraphs');

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for average # of sentences per paragraph
fig.map(sns.kdeplot, 'avg_sents_per_paragraph', shade=True)     .add_legend()     .set(ylim=(0, 0.55));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for average of # words per paragraph
fig.map(sns.kdeplot, 'avg_words_per_paragraph', shade=True)     .add_legend()     .set(ylim=(0, 0.025));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of images
fig.map(sns.kdeplot, 'num_images', shade=True)     .add_legend()     .set(ylim=(0, 0.15));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of non-YouTube videos
fig.map(sns.distplot, 'num_videos', kde=False)     .add_legend()     .set(xlim=(0, 5), xlabel='# of videos');

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of YouTube videos
fig.map(sns.distplot, 'num_youtubes', kde=False)     .add_legend()     .set(xlim=(0, 5), xlabel='# of Youtubes');

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of GIFs
fig.map(sns.distplot, 'num_gifs', kde=False)     .add_legend()     .set(xlim=(0, 5), xlabel='# of GIFs');

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of hyperlinks
fig.map(sns.kdeplot, 'num_hyperlinks', shade=True)     .add_legend()     .set(ylim=(0, 0.13));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for # of bolded text tags
fig.map(sns.kdeplot, 'num_bolded', shade=True)     .add_legend()     .set(ylim=(0, 0.05));

# Setup FacetGrid object and facet on funded status
fig = sns.FacetGrid(renamed_df, hue=' ', size=5, palette='Set1')

# Plot a kde plot for % of bolded text
fig.map(sns.kdeplot, 'percent_bolded', shade=True)     .add_legend()     .set(ylim=(0, 110));

