from owslib.iso import namespaces

# Append gmi namespace to namespaces dictionary.
namespaces.update({'gmi': 'http://www.isotc211.org/2005/gmi'})
namespaces.update({'gml': 'http://www.opengis.net/gml/3.2'})

# Select RA

RAs = {
    'GLOS': 'Great Lakes Observing System',
    'SCCOOS': 'Southern California Coastal Ocean Observing System',
    'SECOORA': 'Southeast Coastal Ocean Observing Regional Association',
    'PacIOOS': 'Pacific Islands Ocean Observing System',
    'NANOOS': 'Northwest Association of Networked Ocean Observing Systems',
}

ra = RAs['SCCOOS']

try:
    from urllib.parse import quote
except ImportError:
    from urllib import quote

# Generate geoportal query and georss feed.

# Base geoportal url.
baseurl = (
    'https://data.nodc.noaa.gov/'
    'geoportal/rest/find/document'
    '?searchText='
)

# Identify the project.
project = (
    'dataThemeprojects:'
    '"Integrated Ocean Observing System '
    'Data Assembly Centers Data Stewardship Program"'
)

# Identify the Regional Association
ra = ' AND "{}" '.format(ra)

# Identify the platform.
platform = 'AND "FIXED PLATFORM"'

# Identify the amount of records and format of the response: 1 to 1010 records.
records = '&start=1&max=1010'

# Identify the format of the response: georss.
response_format = '&f=georss'

# Combine the URL.
url = '{}{}'.format(baseurl, quote(project + ra + platform) + records + response_format)

print('Identified response format:\n{}'.format(url))
print('\nSearch page response:\n{}'.format(url.replace(response_format, '&f=searchPage')))

# Query the NCEI Geoportal and parse the georss response.

from lxml import etree

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

f = urlopen(url)  # Open georss response.
url_string = f.read()  # Read response into string.

# Create etree object from georss response.
url_root = etree.fromstring(url_string)
# Find all iso record links.
iso_record = url_root.findall('channel/item/link')
print('Found %i records' % len(iso_record))
for item in iso_record:
    print(item.text)  # URL to ISO19115-2 record.

# Process each iso record.
get_ipython().magic('matplotlib inline')

from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from owslib import util

fig, ax = plt.subplots(figsize=(15, 12))

i = 0
accenos = []

# For each accession in geo.rss response.
for item in iso_record:
    # Opens the iso xml web reference.
    iso_url = urlopen(item.text)
    # Creates tree element.
    iso_tree = etree.ElementTree(file=urlopen(item.text))
    # Gets the root from tree element.
    root = iso_tree.getroot()
    # Pulls out identifier string.
    ident = root.find(
        util.nspath_eval(
            'gmd:fileIdentifier/gco:CharacterString',
            namespaces
        )
    )
    # Pulls out 7-digit accession number from identifier.
    acce = re.search('[0-9]{7}', util.testXMLValue(ident))
    # Adds accession number to accenos list.
    accenos.append(acce.group(0))
    print('Accession Number = %s' % acce.group(0))

    # Collect Publication date information.
    date_path = (
        'gmd:identificationInfo/'
        'gmd:MD_DataIdentification/'
        'gmd:citation/'
        'gmd:CI_Citation/'
        'gmd:date/'
        'gmd:CI_Date/'
        'gmd:date/gco:Date'
    )
    # First published date.
    pubdate = root.find(util.nspath_eval(date_path, namespaces))
    print('First published date = %s' % util.testXMLValue(pubdate))

    # Collect Provider Platform Codes (if it has it).
    for tag in root.getiterator(
        util.nspath_eval('gco:CharacterString', namespaces)
    ):
        if tag.text == 'Provider Platform Codes':
            # Backs up to the MD_keywords element.
            node = tag.getparent().getparent().getparent().getparent()
            for item in node.findall(
                util.nspath_eval(
                    'gmd:keyword/gco:CharacterString', namespaces
                )
            ):
                print('Provider Platform Code = %s' % item.text)

    # Pull out the version information.
    # Iterate through each processing step which is an NCEI version.
    for tag in root.getiterator(
        util.nspath_eval('gmd:processStep', namespaces)
    ):
        # Only parse gco:DateTime and gmd:title/gco:CharacterString.
        vers_title = (
            'gmi:LE_ProcessStep/'
            'gmi:output/'
            'gmi:LE_Source/'
            'gmd:sourceCitation/'
            'gmd:CI_Citation/'
            'gmd:title/gco:CharacterString'
        )
        vers_date = (
            'gmi:LE_ProcessStep/'
            'gmd:dateTime/gco:DateTime'
        )
        if (
            tag.findall(util.nspath_eval(vers_date, namespaces)) and
            tag.findall(util.nspath_eval(vers_title, namespaces))
        ):
            # Extract dateTime for each version.
            datetimes = tag.findall(util.nspath_eval(vers_date, namespaces))
            # Extract title string (contains version number).
            titles = tag.findall(util.nspath_eval(vers_title, namespaces))
            print('{} = '.format(util.testXMLValue(titles[0]),
                                 util.testXMLValue(datetimes[0])))

    # Collect package size information.
    # Iterate through transfersize nodes.
    for tag in root.getiterator(
        util.nspath_eval('gmd:transferSize', namespaces)
    ):
        # Only go into first gco:Real (where size exists).
        if tag.find(
            util.nspath_eval('gco:Real', namespaces)
        ).text:
            # Extract size.
            sizes = tag.find(util.nspath_eval('gco:Real', namespaces))
            print('Current AIP Size = %s MB' % sizes.text)
            break
        # Only use first size instance, all gco:Real attributes are the same.
        break

    # Bounding time for AIP.
    for tag in root.getiterator(
        util.nspath_eval('gml:TimePeriod', namespaces)
    ):
        # If text exists in begin or end position nodes.
        if (
            tag.find(util.nspath_eval('gml:beginPosition', namespaces)).text and
            tag.find(util.nspath_eval('gml:endPosition', namespaces)).text
        ):
            start_date = tag.find(
                util.nspath_eval('gml:beginPosition', namespaces)
            ).text
            end_date = tag.find(
                util.nspath_eval('gml:endPosition', namespaces)
            ).text
    print('Bounding Time = %s TO %s\n' % (start_date, end_date))

    # Plotting routine for each accession, plot start-end as timeseries for each accession.
    # Create datetime objects for start_date and end_date.
    date1 = datetime(
        int(start_date.split('-')[0]),
        int(start_date.split('-')[1]),
        int(start_date.split('-')[2])
    )
    date2 = datetime(
        int(end_date.split('-')[0]),
        int(end_date.split('-')[1]),
        int(end_date.split('-')[2])
    )
    dates = [date1, date2]
    i += 1  # Counter for plotting.
    y = [i, i]
    # Plot the timeseries.
    ax.plot_date(x=dates, y=y, fmt='-', color='b', linewidth=6.0)

# Clean up the plot.
ax.set_ylim([0, i + 1])
years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsFmt = mdates.DateFormatter('%Y')
ax.xaxis.grid(True)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)  # Format the xaxis labels.
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.grid(True)
ax.set(yticks=np.arange(1, len(accenos)+1))
ax.tick_params(which='both', direction='out')
ax.set_yticklabels(accenos)
plt.ylabel('NCEI Accession Number')
title = ax.set_title('%s Data Archived at NCEI' % ra)

