JACOCO_CSV_FILE = r'input/spring-petclinic/jacoco.csv'

with open (JACOCO_CSV_FILE) as log:
    [print(line, end='') for line in log.readlines()[:4]]

import pandas as pd
coverage= pd.read_csv(JACOCO_CSV_FILE)
coverage.head(3)

coverage = coverage[['PACKAGE', 'CLASS', 'LINE_MISSED', 'LINE_COVERED']]
coverage.head()

coverage['line_size'] = coverage['LINE_MISSED'] + coverage['LINE_COVERED']
coverage['line_covered_ratio'] =  coverage['LINE_COVERED'] / coverage['line_size']
coverage.head()

features = ['Owner', 'Pet', 'Visit', 'Vet', 'Specialty', 'Clinic']

for feature in features:
    coverage.ix[coverage['CLASS'].str.contains(feature), 'feature'] = feature

coverage.ix[coverage['feature'].isnull(), 'feature'] = "Framework"

coverage[['CLASS', 'feature']].head()

feature_usage = coverage.groupby('feature').mean().sort_values(by='line_covered_ratio')[['line_covered_ratio']]
feature_usage

classes_to_delete_by_feature = coverage[coverage['feature'] == feature_usage.index[0]][['PACKAGE', 'CLASS', 'line_covered_ratio', 'line_size']]
classes_to_delete_by_feature

classes_to_delete_by_feature['line_size'].sum() / coverage['line_size'].sum()

coverage['technology'] = coverage['PACKAGE'].str.split(".").str.get(-1)
coverage[['PACKAGE', 'technology']].head()

technology_usage = coverage.groupby('technology').mean().sort_values(by='line_covered_ratio')[['line_covered_ratio']]
technology_usage

classes_to_delete_by_technology = coverage[coverage['technology'] == technology_usage.index[0]][['PACKAGE', 'CLASS', 'line_covered_ratio', 'line_size']]
classes_to_delete_by_technology

classes_to_delete_by_technology['line_size'].sum() / coverage['line_size'].sum()

print("{:.0%}".format(
    (classes_to_delete_by_feature['line_size'].sum() + 
     classes_to_delete_by_technology['line_size'].sum()) / 
     coverage['line_size'].sum()))

import matplotlib.cm as cm
import matplotlib.colors

def assign_rgb_color(value):
    color_code = cm.coolwarm(value)
    return matplotlib.colors.rgb2hex(color_code) 

plot_data = coverage.copy()
plot_data['color'] = plot_data['line_covered_ratio'].apply(assign_rgb_color)
plot_data[['line_covered_ratio', 'color']].head(5)

import json

def create_flare_json(data, 
                      column_name_with_hierarchical_data, 
                      separator=".", 
                      name_column="name", 
                      size_column="size",
                      color_column="color"):
    
    json_data = {}
    json_data['name'] = 'flare'
    json_data['children'] = []

    for row in data.iterrows():
        series = row[1]
        hierarchical_data = series[column_name_with_hierarchical_data]

        last_children = None
        children = json_data['children']

        for part in hierarchical_data.split(separator):
            entry = None

            # build up the tree
            for child in children:
                if "name" in child and child["name"] == part:
                    entry = child
            if not entry:
                entry = {}
                children.append(entry)
            
            # create a new entry section
            entry['name'] = part
            if not 'children' in entry: 
                entry['children'] = []
            children = entry['children']
        
            last_children = children

        # add data to leaf node
        last_children.append({ 
            'name' : series[name_column], 
            'size' : series[size_column],
            'color' : series[color_column]
        })
        
    return json_data
    
json_data = create_flare_json(plot_data, "PACKAGE", ".", "CLASS", "line_size")

print(json.dumps(json_data, indent=3)[0:1000])

FLARE_JSON_FILE = r'vis/flare.json'

with open (FLARE_JSON_FILE, mode='w', encoding='utf-8') as json_file:
    json_file.write(json.dumps(json_data, indent=3))

