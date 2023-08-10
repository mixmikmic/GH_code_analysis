import sys
sys.path.append('..')

import logging
import matplotlib.pyplot as plt
import pandas as pd
import sample_data

from utils import load_data_util

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_cell_magic('time', '', '# prevent printing of validation warning messages for cleaner output\nlogging.getLogger("validator").setLevel(logging.ERROR)\n\nfiles_to_analyze = 10000\ndata = load_data_util.load_random_data(files_to_analyze, False, 42, False)')

get_ipython().run_cell_magic('time', '', '# prevent printing of validation warning messages for cleaner output\nlogging.getLogger("validator").setLevel(logging.ERROR)\n\nsamples_files_to_analyze = 5000\nseeds = [-12, 0, 90, 6258, 73, 1111, 20000, 7512, 2222, 4567]\nsamples = [\n    load_data_util.load_random_data(samples_files_to_analyze, False, seed, False, False) \n    for seed in seeds\n]')

# retrieve the entity list from github and store it in a json object.
entity_list = sample_data.make_entity_list_df()

def get_top_level_domain_entity(url):
    """Returns the domain entity that the url belongs to if it can be found in the disconnectme entity list. 
    If the url cannot be found in the entity list then the empty string is returned. 
    url -- a url for a webpage. Ex. "https://github.com/issues"
    """
    for index, row in entity_list.iterrows():
        for site in row["resources"]:
            match_found = sample_data.find_url_match(url, site)
            if match_found and match_found != "IrregularUrl":
                return row.name
            elif match_found == "IrregularUrl":
                # Not possible to find match for irregular url, so return.
                return ""

    # no match could be found to return the url.
    return ""

def process_results(data_frame):
    """Processes the data retrieved from S3. Returns a JSON object with webpage urls 
    as keys and the processed results as values.
    data_frame -- A Pandas DataFrame that contains S3 JSON data from multiple webpages.
    """
    result = {}
    for index, row in data_frame.iterrows():
        # get the url of the webpage that was being crawled and use thafat as a unique key.
        key = row['location']

        if key not in result:
            result[key] = {
                "domain entity": get_top_level_domain_entity(key),
                "count": 0,
                "total_func_calls": 0,
                "script_urls": {}
            }

        if row['script_loc_eval'] != "":
            result[key]["count"] += 1

            script_url = row['script_url']
            result[key]["script_urls"][script_url] = result[key]["script_urls"].get(script_url, 1) + 1
        result[key]["total_func_calls"] += 1
        
    return result

result = process_results(data)

sample_results = [process_results(sample_data) for sample_data in samples]

class EvalAnalyzer:
    """Analyzes data from processed webpage data and stores the analysis for later use.
    """
    def __init__(self, result):
        """Initializes the objects variables, along with Analyzing the 
        result object and storing that analysis data for later use.
        result -- JSON object with processed S3 webpage data.
        """
        self.webpages_with_eval_calls = 0
        self.external_script_count = 0
        self.external_domain_entity_count = 0
        self.script_urls = []
        self.eval_using_script_count = 0
        self.total_function_calls = 0
        self.webpages_analyzed = len(result)
        self.eval_usage = pd.DataFrame(columns=['url', '# of eval calls', '% of eval calls'])
        
        self.analyze(result)
        
    def analyze(self, result):
        """Analyzes the result object and stores the analysis data in member variables for later use.
        result -- JSON object with processed S3 webpage data.
        """
        for key in result:
            self.total_function_calls += result[key]['total_func_calls']
            if result[key]['count'] > 0:
                self.webpages_with_eval_calls += 1

                self.eval_call_count = result[key]['count']
                self.eval_call_percent = round(result[key]['count'] / result[key]['total_func_calls'] * 100, 2)

                self.eval_usage = self.eval_usage.append({
                    "url": key,
                    "# of eval calls": self.eval_call_count,
                    "% of eval calls": self.eval_call_percent
                }, ignore_index=True)

                for script_url in result[key]["script_urls"]:
                    self.script_urls.append(script_url)
                    self.eval_using_script_count += result[key]["script_urls"][script_url]
                    
                    location_domain_entity = result[key]["domain entity"] if result[key]["domain entity"] != "" else key
                    script_domain_entity = get_top_level_domain_entity(script_url)
                    
                    if (location_domain_entity != script_domain_entity) and (not sample_data.find_url_match(key, script_url)):
                        self.external_domain_entity_count += result[key]["script_urls"][script_url]
                    if not sample_data.find_url_match(key, script_url):
                        self.external_script_count += result[key]["script_urls"][script_url]

        # set '# of eval calls' to be int type since it was being set to object type.
        # without this change later analysis and visualizations do not work.
        self.eval_usage['# of eval calls'] = self.eval_usage['# of eval calls'].astype("int")

analysis = EvalAnalyzer(result)

sample_analyses = [EvalAnalyzer(sample) for sample in sample_results]

function_calls_created_with_eval_percentage = round(analysis.eval_using_script_count / analysis.total_function_calls * 100, 2)

print(
    str(function_calls_created_with_eval_percentage) + "% (" +
    str(analysis.eval_using_script_count) + "/" + str(analysis.total_function_calls) +
    ") of total function calls are created using eval."
)

def get_percentage_of_webpages_with_eval_calls(analysis):
    """Returns the percentage of total webpages in the analysis that have 1 or more eval call.
    analysis -- EvalAnalysis object created by analyzing multiple webpages.
    """
    return round(analysis.webpages_with_eval_calls / analysis.webpages_analyzed * 100, 2)

def print_percentage_of_webpages_with_eval_calls(analysis):
    """Prints the % and number of webpages with 1 or more eval call.
    analysis -- EvalAnalysis object created by analyzing multiple webpages.
    """
    percentage_of_webpages_with_eval_calls = get_percentage_of_webpages_with_eval_calls(analysis)

    print(
        str(percentage_of_webpages_with_eval_calls) + "% (" + 
        str(analysis.webpages_with_eval_calls) + "/" + str(analysis.webpages_analyzed) + 
        ") of webpages have 1 or more function that is created using eval."
    )

print_percentage_of_webpages_with_eval_calls(analysis)

percentage_of_eval_webpages_with_eval_calls_in_samples = pd.Series([
    get_percentage_of_webpages_with_eval_calls(analysis) 
    for analysis in sample_analyses
])

sample_percent_mean = round(percentage_of_eval_webpages_with_eval_calls_in_samples.mean(), 2)
sample_percent_difference = abs(get_percentage_of_webpages_with_eval_calls(analysis) - sample_percent_mean)
sample_percent_variance = round(percentage_of_eval_webpages_with_eval_calls_in_samples.var(), 2)
sample_percent_standard_deviation = round(percentage_of_eval_webpages_with_eval_calls_in_samples.std(), 2)
    
print(str(sample_percent_mean) + "% is the mean % of webpages with 1 or more function created using eval.")
print(str(sample_percent_difference) + 
    "% is the difference between our single large sample and the mean of multiple smaller samples."
)
print(str(sample_percent_variance) + "% is the variance of the % of webpages with 1 or more function created using eval.")
print(str(sample_percent_standard_deviation) + 
    "% is the standard deviation of the % of webpages with 1 or more function created using eval."
)

def get_percentage(numerator, denominator):
    """Returns the percentage value of the numerator divided by the denominator, rounded to 2 decimal places.  
    numerator -- the top value in a percentage calculation.
    denominator -- the bottom value in a percentage calculation.
    """
    return round(numerator / denominator * 100, 2)

def get_number_of_script_urls(analysis):
    """Returns the number of script urls in the anaylsis.
    analysis -- EvalAnalysis object created by analyzing multiple webpages.
    """
    return analysis.eval_using_script_count

def get_number_of_unique_script_urls(analysis):
    """Returns the number of unique script urls in the analysis.
    analysis -- EvalAnalysis object created by analyzing multiple webpages.
    """
    unique_script_urls = set(analysis.script_urls)
    return len(unique_script_urls)

def get_script_domain_entities(analysis):
    """Returns a list of the top level domain entities for each script urls in the analysis.
    The top level domain and retrieved from the disconnectme entity list.
    analysis -- EvalAnalysis object created by analyzing multiple webpages.
    """
    script_domain_entities = []
    
    for url in set(analysis.script_urls):
        domain_entity = get_top_level_domain_entity(url)
        if domain_entity != "":
            script_domain_entities.append(domain_entity)
            
    return script_domain_entities

def get_number_of_script_domain_entities(script_domain_entities):
    """Returns the number of script domain entities in the array.
    script_domain_entities -- an array with domain entity values in it retrieved from the disconnectme entity list.
    """
    return len(script_domain_entities)

def get_number_of_unique_script_domain_entities(script_domain_entities):
    """Returns the number of unique script domain entities in the array.
    script_domain_entities -- an array with domain entity values in it retrieved from the disconnectme entity list.
    """
    unique_script_domain_entities = set(script_domain_entities)
    return len(unique_script_domain_entities)

number_of_script_urls = get_number_of_script_urls(analysis)
number_of_unique_script_urls = get_number_of_unique_script_urls(analysis)

script_domain_entities = get_script_domain_entities(analysis)

number_of_script_domain_entities = get_number_of_script_domain_entities(script_domain_entities)        
number_of_unique_script_domain_entities = get_number_of_unique_script_domain_entities(script_domain_entities)

# calculate the percentages
percentage_of_unique_script_urls = get_percentage(number_of_unique_script_urls, number_of_script_urls)
percentage_of_script_domain_entities = get_percentage(number_of_script_domain_entities, number_of_unique_script_urls)
percentage_of_unqiue_script_domain_entities = get_percentage(
    number_of_unique_script_domain_entities, 
    number_of_script_domain_entities
)

print(str(number_of_script_urls) + " function calls are created using eval in this sample.")

print(
    str(percentage_of_unique_script_urls) + "% (" + 
    str(number_of_unique_script_urls) + "/" + str(number_of_script_urls) + 
    ") of the scripts that use eval are unique."
)

print(
    str(percentage_of_script_domain_entities) + "% (" + 
    str(number_of_script_domain_entities) + "/" + str(number_of_unique_script_urls) + 
    ") of those unqique scripts are part of a domain entity in the disconnectme entity list."
)

print(
    str(percentage_of_unqiue_script_domain_entities) + "% (" + 
    str(number_of_unique_script_domain_entities) + "/" + str(number_of_script_domain_entities) + 
    ") of those domain entities are unique."
)

mean_number_of_script_urls = pd.Series([
        get_number_of_script_urls(sample_analysis) 
        for sample_analysis in sample_analyses
]).mean()

# get the # and % of script urls that are unique
mean_number_of_unique_script_urls = pd.Series([
        get_number_of_unique_script_urls(sample_analysis) 
        for sample_analysis in sample_analyses 
]).mean()

mean_percentage_of_unique_script_urls = round(pd.Series([
        get_percentage(get_number_of_unique_script_urls(sample_analysis), get_number_of_script_urls(sample_analysis))
        for sample_analysis in sample_analyses 
]).mean(), 2)

# get a list of script domain entities to be used for later calculations
sample_script_domain_entities = [
        get_script_domain_entities(sample_analysis)
        for sample_analysis in sample_analyses 
]

# get the mean # and % of script urls that correspond to a domain entity
mean_number_of_script_domain_entities = pd.Series([
        get_number_of_script_domain_entities(domain_entity)
        for domain_entity in sample_script_domain_entities
]).mean()

mean_percentage_of_script_domain_entities = round(pd.Series([
        get_percentage(
            get_number_of_script_domain_entities(domain_entity),
            mean_number_of_unique_script_urls
        )
        for domain_entity in sample_script_domain_entities
]).mean(), 2)



# get the mean # and % of script urls that correspond to a unique domain entity
mean_number_of_unique_script_domain_entities = pd.Series([
        get_number_of_unique_script_domain_entities(domain_entity)
        for domain_entity in sample_script_domain_entities
]).mean()

mean_percentage_of_unique_script_domain_entities = round(pd.Series([
        get_percentage(
            get_number_of_unique_script_domain_entities(domain_entity), 
            get_number_of_script_domain_entities(domain_entity)
        )
        for domain_entity in sample_script_domain_entities
]).mean(), 2)

print(str(mean_number_of_script_urls) + " is the mean number of function calls created using eval in the samples.")

print(
    str(mean_percentage_of_unique_script_urls) + "% (" + 
    str(mean_number_of_unique_script_urls) + "/" + str(mean_number_of_script_urls) + 
    ") is the mean number of unique script urls in the sample."
)

print(
    str(mean_percentage_of_script_domain_entities) + "% (" + 
    str(mean_number_of_script_domain_entities) + "/" + str(mean_number_of_unique_script_urls) + 
    ") is the mean number of domain entities in the unique script urls for our samples."
)

print(
    str(mean_percentage_of_unique_script_domain_entities) + "% (" + 
    str(mean_number_of_unique_script_domain_entities) + "/" + str(mean_number_of_script_domain_entities) + 
    ") is the mean number of unique domain entities."
)

total_script_urls = get_number_of_script_urls(analysis)
percentage_of_external_scripts = round(analysis.external_script_count / total_script_urls * 100, 2)

print(
    str(percentage_of_external_scripts) + "% (" + 
    str(analysis.external_script_count) + "/" + str(total_script_urls) + 
    ") of scripts that use eval are hosted on a different domain than the source webpage."
)


percentage_of_external_domain_entity_scripts = round(analysis.external_domain_entity_count / total_script_urls * 100, 2)
print(
    str(percentage_of_external_domain_entity_scripts) + "% (" + 
    str(analysis.external_domain_entity_count) + "/" + str(total_script_urls) + 
    ") of scripts that use eval are part of a different domain entity than the webpage," + 
    " or have a different domain name than the source webpage."
)

standard_deviation = round(analysis.eval_usage['# of eval calls'].std(), 2)
mean = round(analysis.eval_usage['# of eval calls'].mean(), 2)
median = round(analysis.eval_usage['# of eval calls'].median(), 2)

print('median = ' + str(median))
print('standard deviation = ' + str(standard_deviation))
print('average = ' + str(mean))

print(analysis.eval_usage.sort_values(by=['# of eval calls'], ascending=False))

largest_eval_call_count = analysis.eval_usage['# of eval calls'].max()
plt.figure()

fig = analysis.eval_usage['# of eval calls'].plot.hist(bins=largest_eval_call_count)

plt.title("Distribution of # of eval calls")
fig.set_xlabel("# of eval calls")

plt.show()
plt.close()

plt.figure()

fig = analysis.eval_usage['# of eval calls'].plot.hist(alpha=0.5, bins=largest_eval_call_count)

plt.title("Distribution of # of eval calls")
fig.set_xlabel("# of eval calls")
plt.yscale('log')
fig.set_xlim(1, int(standard_deviation * 2))
fig.grid()

plt.show()
plt.close()

plt.figure()

fig = analysis.eval_usage['# of eval calls'].plot.hist(alpha=0.5, bins=largest_eval_call_count)

plt.title("Distribution of # of eval calls")
fig.set_xlabel("# of eval calls")
fig.set_xlim(1, int(standard_deviation / 4))
fig.grid();

plt.show()
plt.close()

plt.figure()

fig = analysis.eval_usage['% of eval calls'].plot.hist(bins=100)

plt.title("Distribution of % of Functions Calls Created using Eval")
fig.set_xlabel("% of eval calls")

plt.show()
plt.close()

plt.figure()

fig = analysis.eval_usage.plot.scatter(x='# of eval calls', y='% of eval calls');

plt.title("# Eval Calls vs. % of Calls Created with Eval")
# set the x and y bounds so that they are slightly less than 1 so that the dots can be seen better.
fig.set_xlim(-20)
fig.set_ylim(-5)

plt.show()
plt.close()

