import h5py
import numpy as np

ohdsi_file_name = "synpuf_inpatient_combined_readmission.hdf5" # Edit this for your file
f5 = h5py.File(ohdsi_file_name, "r")

# Define helper function for joining labels together
def flatten_column_annotations(f5, base_path, abbreviation=None, field_separator="|", first_part=2, second_part=1):
    column_annotations = f5[base_path + "column_annotations"][...]
    number_of_columns = column_annotations.shape[1]
    if abbreviation is not None:
        abbreviation = field_separator + abbreviation
    else:
        abbreviation = ""
    flattened_list = [column_annotations[first_part, i] + field_separator 
                      + column_annotations[second_part, i] + abbreviation 
                      for i in range(number_of_columns)]
    
    cleaned_flattened_list = []
    for name in flattened_list:
        if name[-1] == field_separator:
            name = name.strip()[:-1]
        
        cleaned_flattened_list += [name]
    
    return np.array(cleaned_flattened_list, dtype=column_annotations.dtype)

# Define paths to data in HDF5 container
condition_path = "/ohdsi/condition_occurrence/"
procedure_path = "/ohdsi/procedure_occurrence/"
person_path = "/ohdsi/person/"
visit_occurrence_path = "/ohdsi/visit_occurrence/"
measurement_path = "/ohdsi/measurement/count/"
observation_path = "/ohdsi/observation/count/"
readmission_30_day_path = "/computed/next/30_days/visit_occurrence/"
past_readmission_30_day_history_path = "/computed/past_history/180/computed/next/30_days/visit_occurrence/"

condition_names = flatten_column_annotations(f5, condition_path, abbreviation="C")
condition_names[0:10]

procedure_names = flatten_column_annotations(f5, procedure_path, abbreviation="P")
procedure_names[0:10]

measurement_names = flatten_column_annotations(f5,measurement_path, abbreviation="M", second_part=0)
measurement_names[0:10]

observation_names = flatten_column_annotations(f5, observation_path, abbreviation="O", second_part=0)
observation_names[0:10]

person_names = flatten_column_annotations(f5, person_path, first_part=0)
person_names

visit_names = flatten_column_annotations(f5, visit_occurrence_path, first_part=0)
visit_names

readmission_names = np.array(["30-day inpatient readmission"], dtype=visit_names.dtype)
readmission_names

readmission_history_names = np.array(["Past history of 30-day inpatient readmissions"], dtype=visit_names.dtype)
readmission_history_names

los_names = np.array(["Length of stay in days"], dtype=visit_names.dtype)

# Helper function for finding
def find_positions(names_array, to_find):
    return np.where(names_array == to_find)[0].tolist()

gender_position = find_positions(person_names, "gender_concept_id|8532")

age_in_years_position = find_positions(visit_names, "age_at_visit_start_in_years_int")
visit_start_julian_day_position = find_positions(visit_names, "visit_start_julian_day")
visit_end_julian_day_position = find_positions(visit_names, "visit_end_julian_day")

condition_ap = f5[condition_path + "core_array"]
procedure_ap = f5[procedure_path + "core_array"]
measurement_ap = f5[measurement_path + "core_array"]
observation_ap = f5[observation_path + "core_array"]
visit_occurrence_ap = f5[visit_occurrence_path + "core_array"]
person_ap = f5[person_path + "core_array"]
readmission_30_day_ap = f5[readmission_30_day_path + "core_array"]
past_readmission_30_day_history_ap = f5[past_readmission_30_day_history_path + "core_array"]
(condition_ap.shape, procedure_ap.shape, measurement_ap.shape, observation_ap.shape, visit_occurrence_ap.shape)

number_of_inpatient_stays = condition_ap.shape[0]
number_of_inpatient_stays

# First two positions age, gender, past history of readmission, los in days
offset = 4
number_of_columns = offset + condition_names.shape[0] + procedure_names.shape[0] + measurement_names.shape[0] + observation_names.shape[0]
number_of_columns

hdf5_file_to_write_to = "inpatient_readmission_analysis.hdf5"
w5 = h5py.File(hdf5_file_to_write_to, "w")

independent_array_ds = w5.create_dataset("/independent/core_array", shape=(number_of_inpatient_stays, number_of_columns) 
                                         ,dtype="i", compression="gzip")

independent_array_ds[:, 0] = visit_occurrence_ap[:, age_in_years_position[0]]

independent_array_ds[:, 1] = 1 + (visit_occurrence_ap[:, visit_end_julian_day_position[0]] - visit_occurrence_ap[:, visit_start_julian_day_position[0]])

independent_array_ds[:, 2] = person_ap[:, gender_position[0]]

independent_array_ds[:, 3] = past_readmission_30_day_history_ap[:,0]

independent_array_ds[:, offset:(offset + condition_names.shape[0])] = condition_ap[...]
offset += condition_names.shape[0]

independent_array_ds[:, offset:(offset + procedure_names.shape[0])] = procedure_ap[...]
offset += procedure_names.shape[0]

independent_array_ds[:, offset:(offset + measurement_names.shape[0])] = measurement_ap[...]
offset += measurement_names.shape[0]

independent_array_ds[:, offset:(offset + observation_names.shape[0])] = observation_ap[...]
offset += observation_names.shape[0]

# For non 0 and 1 values set to 1
core_dummy_variables = w5["/independent/core_array"][:, 2:]
core_dummy_variables[core_dummy_variables > 1] = 1

independent_name_array = np.concatenate((visit_names[age_in_years_position],
                                         los_names, 
                                         person_names[gender_position],
                                         readmission_history_names,
                                         condition_names,
                                         procedure_names,
                                         measurement_names,
                                         observation_names), axis=0)  

independent_name_array.shape

independent_name_array_ds = w5.create_dataset("/independent/column_annotations", shape=(number_of_columns,), 
                                              dtype=independent_name_array.dtype)

independent_name_array_ds[...] = independent_name_array[...]

dependent_array_ds = w5.create_dataset("/dependent/core_array", shape=(number_of_inpatient_stays, 1), 
                                       compression="gzip")

dependent_array_ds[...] = readmission_30_day_ap[...]

dependent_array_name_ds = w5.create_dataset("/dependent/column_annotations", shape=(1,),
                                            dtype=readmission_names.dtype)

dependent_array_name_ds[...] = readmission_names[...]

w5.close()

ff5 = h5py.File(hdf5_file_to_write_to, 'r')

list(ff5["/"])

np.sum(ff5["/independent/core_array"][:,1:])

f5.close()
ff5.close()



