# library imports
import pandas as pd
import requests
import pytz

base_url = "http://0.0.0.0:8000"
headers = {"Authorization": "Bearer tokstr"}

url = base_url + "/api/v1/projects/"
projects = requests.get(url, headers=headers).json()

projects

url = base_url + "/api/v1/consumption_metadatas/?summary=True&projects={}".format(projects[0]['id'])
consumption_metadatas = requests.get(url, headers=headers).json()

consumption_metadatas[0]

url = base_url + "/api/v1/consumption_records/?metadata={}".format(consumption_metadatas[0]['id'])
consumption_records = requests.get(url, headers=headers).json()

consumption_records[:3]

url = base_url + "/api/v1/projects/{}/".format(projects[0]['id'])
requests.delete(url, headers=headers)

project_data = pd.read_csv('sample-project-data.csv',
                           parse_dates=['retrofit_start_date', 'retrofit_end_date']).iloc[0]

project_data

data = {
    "project_id": project_data.project_id,
    "zipcode": str(project_data.zipcode),
    "baseline_period_end": pytz.UTC.localize(project_data.retrofit_start_date).isoformat(),
    "reporting_period_start": pytz.UTC.localize(project_data.retrofit_end_date).isoformat(),
    "project_owner": 1,
}
print(data)

url = base_url + "/api/v1/projects/"
new_project = requests.post(url, json=data, headers=headers).json()
new_project

url = base_url + "/api/v1/projects/"
requests.post(url, json=data, headers=headers).json()

data = [
    {
        "project_id": project_data.project_id,
        "zipcode": str(project_data.zipcode),
        "baseline_period_end": pytz.UTC.localize(project_data.retrofit_start_date).isoformat(),
        "reporting_period_start": pytz.UTC.localize(project_data.retrofit_end_date).isoformat(),
        "project_owner_id": 1,
    }
]
print(data)

url = base_url + "/api/v1/projects/sync/"
requests.post(url, json=data, headers=headers).json()

energy_data = pd.read_csv('sample-energy-data_project-ABC_zipcode-50321.csv',
                          parse_dates=['date'], dtype={'zipcode': str})
energy_data.head()

interpretation_mapping = {"electricity": "E_C_S"}
data = [
    {
        "project_project_id": energy_data.iloc[0]["project_id"],
        "interpretation": interpretation_mapping[energy_data.iloc[0]["fuel"]],
        "unit": energy_data.iloc[0]["unit"].upper(),
        "label": energy_data.iloc[0]["trace_id"].upper()
    }
]
data

url = base_url + "/api/v1/consumption_metadatas/sync/"
consumption_metadatas = requests.post(url, json=data, headers=headers).json()

consumption_metadatas

data = [{
    "metadata_id": consumption_metadatas[0]['id'],
    "start": pytz.UTC.localize(row.date.to_datetime()).isoformat(),
    "value": row.value,
    "estimated": row.estimated,
} for _, row in energy_data.iterrows()]
data[:3]

url = base_url + "/api/v1/consumption_records/sync2/"
consumption_records = requests.post(url, json=data, headers=headers)

consumption_records.text

url = base_url + "/api/v1/consumption_records/?metadata={}".format(consumption_metadatas[0]['id'])
consumption_records = requests.get(url, json=data, headers=headers).json()

consumption_records[:3]

data = {
    "project": new_project['id'],
    "meter_class": "EnergyEfficiencyMeter",
    "meter_settings": {}
}
data

url = base_url + "/api/v1/project_runs/"
project_run = requests.post(url, json=data, headers=headers).json()
project_run

url = base_url + "/api/v1/project_runs/{}/".format(project_run['id'])
project_runs = requests.get(url, headers=headers).json()
project_runs

url = base_url + "/api/v1/project_results/"
project_results = requests.get(url, headers=headers).json()
project_results

