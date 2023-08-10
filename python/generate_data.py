import datetime
import itertools
import random

import names
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine


connection_string = 'postgres://localhost:5432/VincentLa'
engine = create_engine(connection_string)

SCHEMA_NAME = 'tutorial_data_ingest'
engine.execute('CREATE SCHEMA IF NOT EXISTS ' + SCHEMA_NAME)

def create_patients():
    """Creating a table of patient and ids"""
    ids = list(range(1, 11))
    doctor_ids = ['dr' + str((i % 2) + 1) for i in ids]
    names = ['john', 'jeremy', 'mark', 'leslie', 'sam', 'matt', 'judy', 'parth', 'kevin', 'joshua']

    patients = {
        'patient_id': ids,
        'doctor_id': doctor_ids,
        'name': names
    }

    pd.DataFrame(patients).to_sql('patients', engine, schema=SCHEMA_NAME, index=False, if_exists='replace')

def create_risk_assessment_scores():
    """Creating a table of risk_assessments and scores"""
    scores = [
        (1, 'psychological', 100, datetime.date(2017, 1, 5)),
        (2, 'psychological', 96, datetime.date(2017, 1, 5)),
        (3, 'psychological', 89, datetime.date(2017, 1, 5)),
        (4, 'psychological', 75, datetime.date(2017, 1, 5)),
        (5, 'psychological', 81, datetime.date(2017, 1, 5)),
        (1, 'psychological', 90, datetime.date(2017, 1, 12)),
        (1, 'psychological', 92, datetime.date(2017, 1, 12)),
        (3, 'psychological', 94, datetime.date(2017, 1, 13)),
        (4, 'psychological', 85, datetime.date(2017, 1, 15)),
        (6, 'physical', 92, datetime.date(2017, 1, 7)),
        (7, 'physical', 85, datetime.date(2017, 1, 7)),
        (8, 'physical', 72, datetime.date(2017, 1, 7)),
        (9, 'physical', 73, datetime.date(2017, 1, 7)),
        (6, 'physical', 93, datetime.date(2017, 1, 8)),
        (6, 'physical', 94, datetime.date(2017, 1, 9)),
        (7, 'physical', 82, datetime.date(2017, 1, 15)),
        (8, 'physical', 75, datetime.date(2017, 1, 16)),
    ]
    labels = ['patient_id', 'assessment_type', 'risk_score', 'date_modified']
    pd.DataFrame.from_records(scores, columns=labels)        .to_sql('risk_assessments', engine, schema=SCHEMA_NAME, index=False, if_exists='replace')

def create_doctors():
    """Creating a table of doctors and ids"""
    doctor_ids = list(range(1, 3))
    doctor_ids = ['dr' + str(s) for s in doctor_ids]

    names = ['Dr. Smith', 'Dr. Smith']

    doctors = {
        'doctor_id': doctor_ids,
        'names': names,
    }

    pd.DataFrame(doctors).to_sql('doctors', engine, schema=SCHEMA_NAME, index=False, if_exists='replace')

create_doctors()
create_patients()
create_risk_assessment_scores()

def draw_random_int():
    number = np.random.normal(100, 30)
    if number < 1:
        return 1
    else:
        return int(round(number))

upcoders = [0.05, 0.05, 0.15, 0.15, 0.6]
typicalcoders = [0.05, 0.1, 0.4, 0.4, 0.05]

drs = ['Dr. ' + names.get_full_name() for i in range(0, 1000)]
num_of_encounters = [draw_random_int() for i in range(0, 1000)]
personid = [random.randint(1, 25000) for i in range(0, sum(num_of_encounters))]
procedure_map = {
    1: 'Evaluation and Management, Lowest Intensity',
    2: 'Evaluation and Management, Second Lowest Intensity',
    3: 'Evaluation and Management, Medium Intensity',
    4: 'Evaluation and Management, High Intensity',
    5: 'Evaluation and Management, Highest Intensity',
}

doctors = list(itertools.chain(*[[drs[i]] * num_of_encounters[i] for i in range(0, len(num_of_encounters))]))

d = {
    'servicing_provider_npi': doctors,
    'personid': personid,
}
df = pd.DataFrame(d)

procedure_codes = []
for i in df.index:
    if drs.index(df.loc[i, 'servicing_provider_npi']) % 10 < 2:
        procedure_codes.append(np.random.choice(np.arange(1, 6), p=upcoders))
    else:
        procedure_codes.append(np.random.choice(np.arange(1, 6), p=typicalcoders))
procedure_codes = np.asarray(procedure_codes)

df['procedure_code'] = procedure_codes
df['procedure_name'] = df['procedure_code'].map(procedure_map)

df.head()

df.to_sql('claim_lines', engine, schema=SCHEMA_NAME, index=False, if_exists='replace')

