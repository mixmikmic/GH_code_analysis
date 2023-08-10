import calcbench as cb
import pandas as pd
import datetime
pit_columns = ['CIK', 'calendar_period', 'calendar_year', 'date_reported', 
               'fiscal_period', 'fiscal_year', 'metric', 'revision_number','ticker', 'value']

companies = cb.companies(entire_universe=True).ticker

data = pd.DataFrame()
elapsed_start = datetime.datetime.now()
for company_number, ticker in enumerate(companies):
    start = datetime.datetime.now()
    try:
        pit_data = cb.point_in_time(all_footnotes=True, company_identifiers=[ticker])
    except Exception as e:
        print("Exception getting {0}".format(ticker))
    else:
        if not pit_data.empty:
            data = data.append(pit_data[pit_columns])
        print("done getting {0}, it took {1}, elapsed {2}, company_number {3}, average time {4}".format(ticker, 
                                                                                      datetime.datetime.now() - start, 
                                                                                      datetime.datetime.now() - elapsed_start, 
                                                                                      company_number,
                                                                                        (datetime.datetime.now() - elapsed_start) / (company_number + 1)))

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
daily_updates = cb.point_in_time(all_footnotes=True, update_date=yesterday)[pit_columns]

daily_updates.to_excel("C:\\Users\Andrew Kittredge\Documents\pit_footnotes.xlsx")



