# Import the ArcGIS API for Python
from arcgis import *
from IPython.display import display

# Connect to the web GIS.
gis = GIS('https://python.playground.esri.com/portal', 'arcgis_python', 'amazing_arcgis_123')

# Search for content tagged with 'ALS'
search_als = gis.content.search('tags: ALS', 'Feature Layer')
for item in search_als:
    display(item)

# Display the titles of the items returned by the search.
for item in search_als:
    print(item.title)

# Create a map of California.
map1 = gis.map("State of California, USA")
map1.basemap = 'dark-gray'
display(map1)

# Add the ALS content to the map.
for item in search_als:
    if item.title == 'ALS_Clinics_CA':
        als_clinics = item       
    if item.title == 'ALS_Patients_CA':
        patient_locations = item        
    if item.title == 'ALS_Clinic_City_Candidates_CA':
        city_candidates = item    
    elif item.title == 'ALS_Clinic_90minDriveTime_CA':
        drive_times = item
        
map1.add_layer(drive_times)
map1.add_layer(als_clinics)
map1.add_layer(patient_locations)
map1.add_layer(city_candidates)

# Extract the required input data for the analyses.

# ALS clinic city candidates FeatureSet
city_candidates_fset = city_candidates.layers[0].query()

# ALS patient locations FeatureSet
patient_locations_fset = patient_locations.layers[0].query()

# Display the ALS patients FeatureSet in a pandas Dataframe
patient_locations_sdf = patient_locations_fset.df
patient_locations_sdf.head()

# create the 'weight' column
patient_locations_sdf['weight'] = patient_locations_sdf['num_patients']

# drop the 'num_patients' column
patient_locations_sdf2 = patient_locations_sdf.drop('num_patients', axis=1)

# convert SpatialDataFrame back to a FeatureSet
patient_locations_weighted = patient_locations_sdf2.to_featureset()
patient_locations_weighted.df.head()

# Define a function to display the output analysis results in a map
import time

def visualize_locate_allocate_results(map_widget, solve_locate_allocate_result, zoom_level):
    # The map widget
    m = map_widget
    # The locate-allocate analysis result
    result = solve_locate_allocate_result
    
    # 1. Parse the locate-allocate analysis results
    # Extract the output data from the analysis results
    # Store the output points and lines in pandas dataframes
    demand_df = result.output_demand_points.df
    lines_df = result.output_allocation_lines.df

    # Extract the allocated demand points (patients) data.
    demand_allocated_df = demand_df[demand_df['DemandOID'].isin(lines_df['DemandOID'])]
    demand_allocated_fset = features.FeatureSet.from_dataframe(demand_allocated_df)

    # Extract the un-allocated demand points (patients) data.
    demand_not_allocated_df = demand_df[~demand_df['DemandOID'].isin(lines_df['DemandOID'])]
    demand_not_allocated_fset = features.FeatureSet.from_dataframe(demand_not_allocated_df)

    # Extract the chosen facilities (candidate clinic sites) data.
    facilities_df = result.output_facilities.df[['Name', 'FacilityType', 
                                                 'Weight','DemandCount', 'DemandWeight', 'SHAPE']]
    facilities_chosen_df = facilities_df[facilities_df['FacilityType'] == 3]
    facilities_chosen_fset = features.FeatureSet.from_dataframe(facilities_chosen_df)

    # 2. Define the map symbology
    # Allocation lines
    allocation_line_symbol_1 = {'type': 'esriSLS', 'style': 'esriSLSSolid',
                                'color': [255,255,255,153], 'width': 0.7}

    allocation_line_symbol_2 = {'type': 'esriSLS', 'style': 'esriSLSSolid',
                                'color': [0,255,197,39], 'width': 3}

    allocation_line_symbol_3 = {'type': 'esriSLS', 'style': 'esriSLSSolid',
                                'color': [0,197,255,39], 'width': 5}
    
    allocation_line_symbol_4 = {'type': 'esriSLS', 'style': 'esriSLSSolid',
                                'color': [0,92,230,39], 'width': 7}
    
    # Patient points within 90 minutes drive time to a proposed clinic location.
    allocated_demand_symbol = {'type' : 'esriPMS', 'url' : 'https://maps.esri.com/legends/Firefly/cool/1.png',
                               'contentType' : 'image/png', 'width' : 26, 'height' : 26,
                               'angle' : 0, 'xoffset' : 0, 'yoffset' : 0}

    # Patient points outside of a 90 minutes drive time to a proposed clinic location.
    unallocated_demand_symbol = {'type' : 'esriPMS', 'url' : 'https://maps.esri.com/legends/Firefly/warm/1.png',
                                 'contentType' : 'image/png', 'width' : 19.5, 'height' : 19.5,
                                 'angle' : 0, 'xoffset' : 0, 'yoffset' : 0}

    # Selected clinic
    selected_facilities_symbol = {'type' : 'esriPMS', 'url' : 'https://maps.esri.com/legends/Firefly/ClinicSites.png',
                                  'contentType' : 'image/png', 'width' : 26, 'height' : 26,
                                  'angle' : 0, 'xoffset' : 0, 'yoffset' : 0}   
    
    # 3. Display the analysis results in the map
    # Add a slight delay for drama. 
    time.sleep(1.5)  
    # Display the patient-clinic allocation lines.
    m.draw(shape=result.output_allocation_lines, symbol=allocation_line_symbol_4)
    m.draw(shape=result.output_allocation_lines, symbol=allocation_line_symbol_2)
    m.draw(shape=result.output_allocation_lines, symbol=allocation_line_symbol_1)
 
    # Display the locations of patients within the specified drive time to the selected clinic(s).
    m.draw(shape=demand_allocated_fset, symbol=allocated_demand_symbol)

    # Display the locations of patients outside the specified drive time to the selected clinic(s).
    m.draw(shape = demand_not_allocated_fset, symbol = unallocated_demand_symbol)

    # Display the chosen clinic site.
    m.draw(shape=facilities_chosen_fset, symbol=selected_facilities_symbol)

    # Zoom out to display all of the allocated patients points.
    m.zoom = zoom_level

# Identify the city which has the greatest number of patients within a 90 minute drive time.
result1 = network.analysis.solve_location_allocation(problem_type='Maximize Coverage',
    travel_direction='Demand to Facility',
    number_of_facilities_to_find='1',
    demand_points=patient_locations_weighted,
    facilities=city_candidates_fset,
    measurement_units='Minutes',
    default_measurement_cutoff=90
)
print('Analysis succeeded? {}'.format(result1.solve_succeeded))

# Display the analysis results in a pandas dataframe.
result1.output_facilities.df[['Name', 'FacilityType', 'DemandCount', 'DemandWeight']]

# Display the analysis results in a map.

# Create a map of Visalia, California.
map2 = gis.map('Visalia, CA')
map2.basemap = 'dark-gray'
display(map2)

# Call custom function defined earlier in this notebook to 
# display the analysis results in the map.
visualize_locate_allocate_results(map2, result1, zoom_level=7)

# Identify where to open new clinics that are within a 90 minute drive time
# of 50% of ALS patients who currently have to drive for over 90 minutes to reach their nearest clinic.

result2 = network.analysis.solve_location_allocation(
    problem_type='Target Market Share', 
    target_market_share=70,
    facilities=city_candidates_fset, 
    demand_points=patient_locations_weighted,
    travel_direction='Demand to Facility',
    measurement_units='Minutes', 
    default_measurement_cutoff=90
)

print('Solve succeeded? {}'.format(result2.solve_succeeded))

# Display the analysis results in a table.
result2.output_facilities.df[['Name', 'FacilityType', 'DemandCount', 'DemandWeight']]

# Display the analysis results in a map.

# Create a map of Visalia, California
map3 = gis.map('Visalia, CA', zoomlevel=7)
map3.basemap = 'dark-gray'

# Display the map and add the analysis results to it
from IPython.display import display
display(map3)
visualize_locate_allocate_results(map3, result2, zoom_level=6)

