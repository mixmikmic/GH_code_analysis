from pprint import pprint
import numpy as np
# Let's import the yahoo finance API
import yahoo_finance

def download_data(ticker, start_date, end_date):
    # Do a lookup on a ticker
    share = yahoo_finance.Share(ticker)
    # And query the historical data
    return share.get_historical(start_date, end_date)

# And a little test with Walmart
pprint(download_data('WMT', '2016-02-10', '2016-02-16')[0])

def download_from_yf(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = download_data(ticker, start_date, end_date)
    return data

# We need to tell the downloader what stocks we care about!
tickers = ['WMT', 'XOM'] # Walmart and ExxonMobil from example in class

# And also a timerange, i.e. the last month
import datetime
import dateutil.relativedelta

end_date = datetime.date.today()

# And now we just have to subtract a month from end_date
start_date = end_date - dateutil.relativedelta.relativedelta(months=1)
print "Fetch stock data for %s from"%", ".join(tickers), start_date, "to", end_date


date_format = '%Y-%m-%d' # YYYY-MM-DD in Python
s_date = start_date.strftime(date_format)

# We could have alternatively cast the dates into a
# string since %Y-%m-%d is the the default date format
# s_date = str(start_date)

e_date = end_date.strftime(date_format)

data = download_from_yf(tickers, s_date, e_date)
from pprint import pprint
pprint(data)

def convert_to_weekly(data):
    def __convert_to_weekly(data):
        # Sort the data by time from oldest date to most recent
        data = sorted(data, key=lambda row: datetime.datetime.strptime(row['Date'], date_format))
        weeks = {}
        
        # High level approach: Markets typically open on a Monday,
        # so we see if the Monday of a given week is in the weeks dict
        #   if it is: we aren't the first day the markets were open on a given week
        #     - Update adj. close and close
        #     - Append volume to list
        #     - And check low and high
        #   if it isn't: we are the first day
        #     - Do the above stuff and also set the open price
        
        for row in data:
            d = datetime.datetime.strptime(row['Date'], date_format)
            # Expected start of the week
            d_start = (d-datetime.timedelta(d.weekday())).strftime(date_format)
            if d_start not in weeks:
                # First day in a given week
                weeks[d_start] = row
                weeks[d_start]['Volume'] = [row['Volume']]
            else:
                # Append the daily volume to the weekly count
                weeks[d_start]['Volume'].append(row['Volume'])
                # Update close and adj. close since we are the latest date
                weeks[d_start]['Close'] = row['Close']
                weeks[d_start]['Adj_Close'] = row['Adj_Close']

                # Check if we are higher than the highest observed value
                if float(weeks[d_start]['High']) < float(row['High']):
                    weeks[d_start]['High'] = row['High']

                # Check if we are lower than the lowest observed value
                if float(weeks[d_start]['Low']) > float(row['Low']):
                    weeks[d_start]['Low'] = row['Low']

        rows = []
        for k,v in weeks.items():
            # Sum up the volumes and divide by the count (compute average volume)
            v['Volume'] = np.average(map(int, v['Volume']))
            rows.append(v)
        return rows
        
    for ticker in data.keys():
        data[ticker] = __convert_to_weekly(data[ticker])
    return data

weekly_data = convert_to_weekly(data)
pprint(weekly_data)

# Python's decimal class let's us represent decimals 
# with a fixed number of digits after the decimal place
import decimal
# This is no good as 73.9... is a float and we will represent it to 26 places of accuracy
dec = decimal.Decimal(73.910002)
print dec

# If we round it first, we get nicer results :)
dec = decimal.Decimal(str(round(73.910002, 2)))
print dec

# Let's import it and get started
import xlsxwriter

def write_xlsx(data, output_file):
    header = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
    # Create a new workbook
    workbook = xlsxwriter.Workbook(output_file)
    # Pick the format that you want Excel to use for dates
    # mm/dd/yy is the default, so let's use that
    excel_date_format = workbook.add_format({'num_format': 'mm/dd/yy'})
    
    def __write_val(sheet, field, row, row_pos, col_pos):
        # A little helper func for dealing with different data types
        if field == 'Date':
            # Write the date to cell (col_pos, row_pos) i.e. (A1)
            sheet.write_datetime(
                row_pos,
                col_pos,
                datetime.datetime.strptime(row['Date'], date_format),
                excel_date_format
            )
        elif field == 'Volume':
            # Volume is an int, so we will write it as such
            sheet.write_number(row_pos, col_pos, int(row[field]))
        else:
            sheet.write_number(
                row_pos,
                col_pos,
                # From the aside
                decimal.Decimal(str(round(float(row[field]), 2)))
            )
    # Add the worksheets in sorted order
    for ticker in sorted(data.keys()):
        # Just name the worksheet after the stock
        worksheet = workbook.add_worksheet(ticker)
        row_pos = 0
        col_pos = 0
        # Write the header to the sheet
        for field in header:
            worksheet.write(row_pos, col_pos, field)
            col_pos += 1
        row_pos += 1
        
        # Sort the dates in descending order
        for row in sorted(data[ticker],
            key=lambda r: datetime.datetime.strptime(r['Date'], date_format)):
                col_pos = 0
                for field in header:
                    __write_val(worksheet, field, row, row_pos, col_pos)
                    col_pos += 1
                row_pos += 1

    # And finally close the workbook
    workbook.close()
        
    

# And let's try it
out_file = 'ticker_prices.xlsx'
write_xlsx(weekly_data, out_file)
# Should now have ticker_prices.xlsx in your working directory
import os
# Just a little check to see if the file is there
assert(out_file in os.listdir('.'))

