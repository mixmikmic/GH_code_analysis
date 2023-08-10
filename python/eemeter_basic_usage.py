# library imports
from eemeter.structures import (
    EnergyTrace,
    EnergyTraceSet,
    Intervention,
    ZIPCodeSite,
    Project
)
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter
import pandas as pd
import pytz

energy_data = pd.read_csv('sample-energy-data_project-ABC_zipcode-50321.csv',
                          parse_dates=['date'], dtype={'zipcode': str})
records = [{
    "start": pytz.UTC.localize(row.date.to_datetime()),
    "value": row.value,
    "estimated": row.estimated,
} for _, row in energy_data.iterrows()]

energy_trace = EnergyTrace(
    records=records,
    unit="KWH",
    interpretation="ELECTRICITY_CONSUMPTION_SUPPLIED",
    serializer=ArbitraryStartSerializer())

energy_trace_set = EnergyTraceSet([energy_trace], labels=["DEF"])

project_data = pd.read_csv('sample-project-data.csv',
                           parse_dates=['retrofit_start_date', 'retrofit_end_date']).iloc[0]

retrofit_start_date = pytz.UTC.localize(project_data.retrofit_start_date)
retrofit_end_date = pytz.UTC.localize(project_data.retrofit_end_date)

interventions = [Intervention(retrofit_start_date, retrofit_end_date)]

site = ZIPCodeSite(project_data.zipcode)

project = Project(energy_trace_set=energy_trace_set, interventions=interventions, site=site)

meter = EnergyEfficiencyMeter()
results = meter.evaluate(project)

project_derivatives = results['project_derivatives']

project_derivatives.keys()

modeling_period_set_results = project_derivatives[('baseline', 'reporting')]

modeling_period_set_results.keys()

electricity_consumption_supplied_results = modeling_period_set_results['ELECTRICITY_CONSUMPTION_SUPPLIED']

electricity_consumption_supplied_results.keys()

baseline_results = electricity_consumption_supplied_results["BASELINE"]
reporting_results = electricity_consumption_supplied_results["REPORTING"]

baseline_results.keys()

reporting_results.keys()

baseline_normal = baseline_results['annualized_weather_normal']
reporting_normal = reporting_results['annualized_weather_normal']

percent_savings = (baseline_normal[0] - reporting_normal[0]) / baseline_normal[0]

percent_savings

