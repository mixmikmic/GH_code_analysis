from databaker.framework import *


# FILENAMES - this is the ONLY bit that should need to change
sa_inputfile = "ilchtablestemplatesa.xls"
nsa_inputfile = "ilchtablestemplatensa.xls"

# Shared functions

# Get the growth period
def get_growthPeriod(tab):
    tab_title = tab.excel_ref('A1')
    
    if tab_title.filter(contains_string("year on year")):
        gp = "Annual"
    elif tab_title.filter(contains_string("quarter on quarter")):
        gp = "Quarterly"
    elif tab_title.filter(contains_string("growth rates")):
        gp = "Annual"
    return gp


# Get the measure type
def get_measureType(tab):
    tab_title = tab.excel_ref('A1')
                       
    if tab_title.filter(contains_string("year on year")):
        mt = "Percent"
    elif tab_title.filter(contains_string("quarter on quarter")):
        mt = "Percent"
    elif tab_title.filter(contains_string("growth rates")):
        mt = "Percent"
    else:
        mt = "Index"
    return mt



def growth_recipe(saOrNsa, tabs_growth):
    
    conversionsegments = []

    for tab in tabs_growth:

        # Set anchor one to the left of cell with "Agriculture" 
        anchor = tab.filter(contains_string("eriod")).assert_one()

        # set up a waffle
        datarows = anchor.fill(DOWN).is_not_blank()
        datacols = anchor.shift(DOWN).fill(RIGHT).is_not_blank()
        obs = datarows.waffle(datacols).is_not_blank()

        # set the growth period & measuretype
        gp = get_growthPeriod(tab)
        mt = get_measureType(tab)

        dimensions = [
                HDimConst(MEASURETYPE, mt),
                HDim(datarows, TIME, DIRECTLY, LEFT),
                HDim(datacols.parent(), "Costs", DIRECTLY, ABOVE),
                HDim(anchor.fill(RIGHT).parent(), "SIC", CLOSEST, LEFT),
                HDimConst("Growth Period", gp),
                HDimConst("SA / NSA", saOrNsa)
                     ]

        # TIME has wierd data markings, get them out
        time = dimensions[1]
        assert time.name == 'TIME', "Time needs to be dimension 0"
        for val in time.hbagset:
            if '(r)' in val.value or ('p') in val.value:
                time.cellvalueoverride[val.value] = val.value[:6]

        conversionsegment = ConversionSegment(tab, dimensions, obs)
        conversionsegments.append(conversionsegment)
    
    return conversionsegments


def level_recipe(saOrNsa):
    
    conversionsegments = []

    for tab in tabs_level:

        # Set anchor one to the left of cell with "Agriculture" 
        anchor = tab.filter(contains_string("eriod")).assert_one()

        # set up a waffle
        datarows = anchor.fill(DOWN).is_not_blank()
        datacols = anchor.shift(DOWN).fill(RIGHT).is_not_blank()
        obs = datarows.waffle(datacols).is_not_blank()
        
        # set the measuretype
        mt = get_measureType(tab)

        dimensions = [
                HDim(datarows, TIME, DIRECTLY, LEFT),
                HDim(datacols.parent(), "Costs", DIRECTLY, ABOVE),
                HDim(anchor.fill(RIGHT).parent(), "SIC", CLOSEST, LEFT),
                HDimConst(MEASURETYPE, mt),
                HDimConst("SA / NSA", saOrNsa)
                     ]

        # TIME has wierd data markings, get them out
        time = dimensions[0]
        assert time.name == 'TIME', "Time needs to be dimension 0"
        for val in time.hbagset:
            if '(r)' in val.value or ('p') in val.value:
                time.cellvalueoverride[val.value] = val.value[:6]

        conversionsegment = ConversionSegment(tab, dimensions, obs)
        conversionsegments.append(conversionsegment)
    
    return conversionsegments


tabs = loadxlstabs(nsa_inputfile)


# get the growth and level tabs
tabs_growth = [x for x in tabs if 'growth' in x.name]
tabs_level = [x for x in tabs if 'level' in x.name]

# Sanity check
assert len(tabs_growth) == 2, "We expect the NSA file to have 2 tabs with the word 'growth' in them"
assert len(tabs_level) == 2, "We expect the NSA file require 2 tabs with the word 'level' in them"


# Growth, NSA
outputfile = 'Output-NSA-growth-' + nsa_inputfile[:-4] + '.csv'
writetechnicalCSV(outputfile, growth_recipe("Not seasonally adjusted"))

# LEvel SA
outputfile = 'Output-NSA-level-' + nsa_inputfile[:-4] + '.csv'
writetechnicalCSV(outputfile, level_recipe("Not seasonally adjusted"))

tabs = loadxlstabs(sa_inputfile)


# get the growth and level tabs
tabs_growth = [x for x in tabs if 'growth' in x.name]
tabs_level = [x for x in tabs if 'level' in x.name]

# Sanity check
assert len(tabs_growth) == 4, "We expect the SA file to have 4 tabs with the word 'growth' in them"
assert len(tabs_level) == 2, "We expect the SA file require 2 tabs with the word 'level' in them"


# Growth, SA
outputfile = 'Output-SA-growth-' + sa_inputfile[:-4] + '.csv'
writetechnicalCSV(outputfile, growth_recipe("Seasonally Adjusted"))   # 'A' to match previous months

# LEvel SA
outputfile = 'Output-SA-level-' + sa_inputfile[:-4] + '.csv'
writetechnicalCSV(outputfile, level_recipe("Seasonally adjusted"))

#topandas(level_recipe("Seasonally adjusted")[0])
20/365



