# Standard imports
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import datetime
get_ipython().magic('matplotlib inline')
# EveKit imports
from evekit.reference import Client
from evekit.util import convert_raw_time

# We'll use a day of order book data for all ore and ice types and their refined materials.
# This cell sets up our reference date, region and station.  The next cells derive the appropriate
# inventory types and retrieve appropriate market data.
#
sde_client = Client.SDE.get()
region_query = "{values: ['The Forge']}"
station_query = "{values: ['Jita IV - Moon 4 - Caldari Navy Assembly Plant']}"
region_id = sde_client.Map.getRegions(regionName=region_query).result()[0][0]['regionID']
station_id = sde_client.Station.getStations(stationName=station_query).result()[0][0]['stationID']
compute_date = datetime.datetime(2017, 1, 10)
print("Using region_id=%d, station_id=%d at %s" % (region_id, station_id, str(compute_date)))

# Ore and Ice types are organized by inventory group.  There are 16 ore families and one ice family,
# each corresponding to an inventory group.  We first need to resolve these group IDs, then we
# can retrieve the corresponding types.
material_group_names = [ 'Veldspar', 'Scordite', 'Pyroxeres', 'Plagioclase', 'Omber', 'Kernite', 'Jaspet', 
                         'Hemorphite', 'Hedbergite', 'Gneiss', 'Dark Ochre', 'Spodumain', 'Crokite', 
                         'Bistot', 'Arkonor', 'Mercoxit', 'Ice' ]
group_name_query = "{values:[" + ",".join(map(lambda x : "'" + x + "'", material_group_names)) + "]}"
material_groups = Client.SDE.load_complete(sde_client.Inventory.getGroups, groupName=group_name_query)

# Next, we'll retrieve type information for all the inventory items in the requested groups
group_id_query = "{values:[" + ",".join([str(x['groupID']) for x in material_groups]) + "]}"
source_types = {}
for next_type in Client.SDE.load_complete(sde_client.Inventory.getTypes, groupID=group_id_query):
    if next_type['marketGroupID'] is not None:
        # We perform this check because the 'Ice' family in the SDE includes some non-refinable types
        # These are detectable by the lack of a market group ID.  We create a material_map entry
        # in preparation for the next step.
        next_type['material_map'] = {}
        source_types[next_type['typeID']] = next_type

# Finally, we'll determine the types which each source material can refine to.  We'll save this information
# as a map associated with each source type.
type_id_query = "{values:[" + ",".join([str(x) for x in source_types.keys()]) + "]}"
for next_mat in Client.SDE.load_complete(sde_client.Inventory.getTypeMaterials, typeID=type_id_query):
    source_types[next_mat['typeID']]['material_map'][next_mat['materialTypeID']] = next_mat

#
# WARNING: this cell will take some time to evaluate if your connection is slow
#
# Now we'll download a day of order book data for our source types and all of the refined materials.
# This isn't mandatory and if you'd prefer to load everything online on demand, you can remove the 
# "local_storage" argument from the order book functions below.  However, we strongly recommend 
# downloading order books to local storage to allow you to iterate more easily as you experiment.
download_types = set(source_types.keys())
for next_type in source_types.values():
    download_types = download_types.union(next_type['material_map'].keys())
    
from evekit.online.Download import download_order_book_range
download_order_book_range([compute_date], ".", types=download_types, regions=[region_id], 
                          config={'verbose': True, 'tree': True})

# We'll load order book data for the day into a DataFrame.
from evekit.marketdata import OrderBook
order_book = OrderBook.get_data_frame(dates=[compute_date], types=download_types, regions=[region_id], 
                                      config=dict(local_storage=".", tree=True, skip_missing=True, 
                                                  fill_gaps=True, verbose=True))

# Set efficiency, tax rate and station tax.  These factors depend on players skills,
# refining location, and player standings with the owner of the refinery.

# This is the efficiency at a typical NPC station with max skills
efficiency = 0.5 * 1.15 * 1.1 * 1.1

# This is the sales tax at a typical NPC station with max skills
tax_rate = 0.01

# Station tax can be no greater than 0.05.  This value is zero if standings are 6.67 or better.
# As noted above, we're substituting order price for adjusted price.  From empirical observation,
# setting station_tax to 0.04 roughly approximates a station_tax of 0.05 with true adjusted prices.
# So we'll set station tax to 0.04 here.
station_tax = 0.04

# For our initial implementation of our strategy, we'll focus on just finding the first
# opportunity that exists for a given asset in a given snapshot.  This won't maximize profit
# for the opportunity, but is simpler to implement as a first cut.

# A key part of our strategy involves determing whether it is possible to buy or sell
# assets in a given snapshot.  We'll abstract this functionality into two separate functions.

# The following function checks whether it is possible to buy the given type at
# the given location in the given volume.  If a purchase is possible, then a list
# of the form [ {price=p1, volume=v1}, ..., {price=pn, volume=vn}] is returned where 
# pi gives a price and vi gives the volume that was purchased at that price.
def attempt_buy_type(buy_location_id, buy_type_id, buy_volume, snapshot):
    # Restrict to the given type and location
    by_type = snapshot[snapshot.type_id == buy_type_id]
    by_loc = by_type[by_type.location_id == buy_location_id]
    by_side = by_loc[by_loc.buy == False]
    # Attempt to buy from order list.  Recall that sell orders are ordered
    # low price to high price in the DataFrame.
    buy_orders = []
    for next_row in by_side.iterrows():
        order = next_row[1]
        if buy_volume >= order['min_volume']:
            amount = min(buy_volume, order['volume'])
            buy_orders.append(dict(price=order['price'], volume=amount))
            buy_volume -= amount
        if buy_volume == 0:
            # We've completely filled this order
            return buy_orders
    # If we never completely fill the order then return no orders
    return []

# This next function checks wheter it is possible to sell the given type from
# a given location at the given volume.  As you may recall from example 3, matching buy orders
# may be at remote locations.  We'll use our range checker from the TradingUtil library
# to help check for matches.
from evekit.marketdata import TradingUtil

# The result of this function will be a list [ {price=p1, volume=v1}, ..., {price=pn, volume=vn}] 
# such that pi gives a price, and vi gives the volume that was sold at that price.
def attempt_sell_type(sell_region_id, sell_location_id, sell_type_id, sell_volume, snapshot):
    config = dict(use_citadel=False)
    by_type = snapshot[snapshot.type_id == sell_type_id]
    by_side = by_type[by_type.buy == True]
    sell_orders = []
    # Attempt to sell to order list.  Recall that buy orders are ordered
    # high price to low price in the DataFrame.
    for next_row in by_side.iterrows():
        order = next_row[1]
        try:
            if sell_volume >= order['min_volume'] and                TradingUtil.check_range(sell_region_id, sell_location_id, order['location_id'], order['order_range'], config):
                amount = min(sell_volume, order['volume'])
                sell_orders.append(dict(price=order['price'], volume=amount))
                sell_volume -= amount
        except:
            # We'll get an exception if TradingUtil can't find the location of a player-owned
            # station.  We'll ignore those for now.  Change "use_citadel" to True above
            # if you'd like to attempt to resolve the location of these stations from a 
            # third party source.
            pass
        if sell_volume == 0:
            # We've completely filled this order
            return sell_orders
    # If we never completely fill the order then return no orders
    return []

# Now we can implement our basic opportunity checker.
def check_opportunities(order_book, type_map, station_id, region_id, efficiency, sales_tax, station_tax, verbose=False):
    if verbose:
        total_snapshots = len(order_book.groupby(order_book.index))
        print("Checking %d snapshots for opportunities" % total_snapshots)
    opportunities = []
    count = 0
    #
    # We'll iterate through every snapshot in the current order book
    for snapshot_group in order_book.groupby(order_book.index):
        # Each group is a pair (snapshot_time, snapshot_dataframe)
        snapshot_time = snapshot_group[0]
        snapshot = snapshot_group[1]
        if verbose:
            print("X", end='')
            count += 1
            if count % 72 == 0:
                print()
        #
        # Next we'll iterate through each source type looking for opportunities
        for source_type in type_map.values():
            #
            # First, attempt to buy enough material to refine
            required_volume = source_type['portionSize']
            buy_orders = attempt_buy_type(station_id, source_type['typeID'], required_volume, snapshot)
            if len(buy_orders) == 0:
                # Failed to buy enough, skip this source
                continue
            buy_total = np.sum([x['price'] * x['volume'] for x in buy_orders])
            #
            # Now attempt to sell all the refined materials
            sell_total = 0
            for next_mat in source_type['material_map'].values():
                # Output amount is determined by efficiency.  This is how much we have to sell
                output_amount = int(next_mat['quantity'] * efficiency)
                # Check whether we can sell this output.
                sell_orders = attempt_sell_type(region_id, station_id, next_mat['materialTypeID'], output_amount, snapshot)
                if len(sell_orders) == 0:
                    # We couldn't sell this material so no opportunity here
                    sell_total = 0
                    break
                # Add the profit from the sale less sales tax
                sell_total += (1 - sales_tax) * np.sum([x['price'] * x['volume'] for x in sell_orders])
                # Add the incremental refinement tax to the buy cost.
                # If we had actual adjusted_prices, we'd use those prices in place of x['price'] below.
                buy_total += station_tax * np.sum([x['price'] * x['volume'] for x in sell_orders])
            # Did we profit?
            if sell_total > buy_total:
                # Yes, record the result
                profit = sell_total - buy_total
                margin = profit/buy_total
                opportunities.append(dict(time=snapshot_time, profit=profit, margin=margin, type=source_type['typeName']))
    if verbose:
        print()
    return opportunities

# Now let's check for all opportunities for our day of book data
opportunities = check_opportunities(order_book, source_types, station_id, region_id, 
                                    efficiency, tax_rate, station_tax, verbose=True)

# Dump opportunities in a nice format
def display_opportunities(opps):
    for opp in opps:
        profit = "{:15,.2f}".format(opp['profit'])
        margin = "{:8.2f}".format(opp['margin'] * 100)
        print("ArbOpp time=%s  profit=%s  return=%s%%  type=%s" % (str(opp['time']), profit, margin, opp['type']))
    print("Total opportunities: %d" % len(opps))

# Let's take a look at the opportunities we found
display_opportunities(opportunities)

# There seem to be ample arbitrage opportunities on our sample day.  To capture the full potential 
# of an opportunity, we need to continue to buy and refine source assets until it is no longer
# profitable to do so.  This requires that we track the state of buy and sell orders during
# each round of processing an opportunity.  We can implement this tracking by extracting order
# lists from each snapshot, and updating volume as we consume orders.  We'll implement new
# versions of our buyer and seller functions to handle this tracking.

# The new version of our asset buyer will attempt to buy from a list of orders which are
# assumed to already be filtered to sell orders of the given type and the appropriate
# location.  This function will consume orders to fill the given volume, and will return
# a list of objects {price, volume} showing the orders that were made.  This list will
# be empty if we can not fill the order completely.
def attempt_buy_type_list(buy_volume, sell_order_list):
    potential = []
    for next_order in sell_order_list:
        if buy_volume >= next_order['min_volume'] and next_order['volume'] > 0:
            # Buy into this order
            amount = min(buy_volume, next_order['volume'])
            order_record = dict(price=next_order['price'], volume=amount)
            buy_volume -= amount
            next_order['volume'] -= amount
            potential.append(order_record)
        if buy_volume == 0:
            # We've completely filled this order
            return potential
    # If we never completely fill the order then return no orders
    return []

# The new version of our asset seller will attempt to sell to a list of orders which
# are assumed to already be filtered to buy orders of the given type.  We use our
# range checker to implement proper ranged buy order matching.  This function will 
# consume volume from an order if possible, and return a list of objects {price, volume} 
# showing the orders that were filled.  This list will be empty if we can not fill
# the order completely.
def attempt_sell_type_list(sell_region_id, sell_location_id, sell_volume, buy_order_list):
    config = dict(use_citadel=False)
    potential = []
    for next_order in buy_order_list:
        try:
            if sell_volume >= next_order['min_volume'] and next_order['volume'] > 0 and                TradingUtil.check_range(sell_region_id, sell_location_id, next_order['location_id'], 
                                       next_order['order_range'], config):
                # Sell into this order
                amount = min(sell_volume, next_order['volume'])
                order_record = dict(price=next_order['price'], volume=amount)
                sell_volume -= amount
                next_order['volume'] -= amount
                potential.append(order_record)
        except:
            # We'll get an exception if TradingUtil can't find the location of a player-owned
            # station.  We'll ignore those for now.  Change "use_citadeL" to True above
            # if you'd like to attempt to resolve the location of these stations from a 
            # third party source.
            pass
        if sell_volume == 0:
            # We've completely filled this order
            return potential
    # If we never completely fill the order then return no orders
    return []                    

# We'll include a few other convenience functions to simplify our implementation

# This function extracts sell orders from a snapshot based on type and station ID.
# Recall that sell orders are sorted low price to high price in the DataFrame.
def extract_sell_orders(snapshot, type_id, station_id):
    by_type = snapshot[snapshot.type_id == type_id]
    by_loc = by_type[by_type.location_id == station_id]
    by_side = by_loc[by_loc.buy == False]
    return [next_order[1] for next_order in by_side.iterrows()]

# This function extracts buy orders from a snapshot based on type ID.
# Recall that buy orders are sorted high price to low price in the DataFrame.
def extract_buy_orders(snapshot, type_id):
    by_type = snapshot[snapshot.type_id == type_id]
    by_side = by_type[by_type.buy == True]
    return [next_order[1] for next_order in by_side.iterrows()]

# This function will combine orders by price so that the resulting
# list has one entry for each price, with the total volume filled at
# that price.  This compression simplifies the display of orders in
# our output functions.
def compress_order_list(order_list, ascending=True):
    order_map = {}
    for next_order in order_list:
        if next_order['price'] not in order_map:
            order_map[next_order['price']] = next_order['volume']
        else:
            order_map[next_order['price']] += next_order['volume']
    orders = [ dict(price=k,volume=v) for k, v in order_map.items()]
    return sorted(orders, key=lambda x: x['price'], reverse=not ascending)

# Now we can write a function that attempts to consume all opportunities for a single type
# in a given snapshot.  This function will attempt to buy and refine as long as it is profitable 
# to do so.  The result of this function will be None if no opportunity was available, or an object:
#
# {
#   gross: gross proceeds (total of all sales)
#   cost: total cost (cost of buying source plus refinement costs)
#   profit: gross - cost
#   margin: cost / profit
#   buy_orders: the compressed list of buy orders that were placed
#   sell_orders: map from material type ID to the compressed list of sell orders that were placed
# }
#
# Compressed order lists group orders by price and sum the volume.
#
def attempt_opportunity(snapshot, type_id, region_id, station_id, type_map, tax_rate, efficiency, station_tax):
    # Reduce to type to extract minimum reprocessing volume
    by_type = snapshot[snapshot.type_id == type_id]
    required_volume = type_map[type_id]['portionSize']
    #
    # Create source sell order list.
    sell_order_list = extract_sell_orders(snapshot, type_id, station_id)
    #
    # Create refined materials buy order lists.
    buy_order_map = {}
    all_sell_orders = {}
    for next_mat in type_map[type_id]['material_map'].values():
        mat_type_id = next_mat['materialTypeID']
        buy_order_map[mat_type_id] = extract_buy_orders(snapshot, mat_type_id)
        all_sell_orders[mat_type_id] = []
    #
    # Now iterate through sell orders until we stop making a profit
    all_buy_orders = []
    gross = 0
    cost = 0
    while True:
        #
        # Attempt to buy source material
        current_cost = 0
        current_gross = 0
        bought = attempt_buy_type_list(required_volume, sell_order_list)
        if len(bought) == 0:
            # Can't buy any more source material, done with this opportunity
            break
        #
        # Add cost of buying source material
        current_cost = np.sum([ x['price'] * x['volume'] for x in bought ])
        #
        # Now attempt to refine and sell all refined materials
        sell_orders = {}
        for next_mat_id in buy_order_map.keys():
            sell_volume = int(type_map[type_id]['material_map'][next_mat_id]['quantity'] * efficiency)
            sold = attempt_sell_type_list(region_id, station_id, sell_volume, buy_order_map[next_mat_id])
            if len(sold) == 0:
                # Can't sell any more refined material, done with this opportunity
                sell_orders = []
                break
            #
            # Add gross profit from selling refined material
            current_gross += (1 - tax_rate) * np.sum([ x['price'] * x['volume'] for x in sold ])
            #
            # Add incremental cost of refining source to this refined material.
            # If we had actual adjusted_prices, we'd use those prices in place of x['price'] below.
            current_cost += station_tax * np.sum([ x['price'] * x['volume'] for x in sold ])
            #
            # Save the set of sale orders we just made
            sell_orders[next_mat_id] = sold
        #
        if len(sell_orders) == 0:
            # We couldn't sell all refined material, so we're done with this opportunity
            break
        #
        # Check whether we've made a profit this round.  If so, record the amounts and continue
        if current_gross > current_cost:
            all_buy_orders.extend(bought)
            for i in sell_orders.keys():
                all_sell_orders[i].extend(sell_orders[i])
            cost += current_cost
            gross += current_gross
        else:
            # This round didn't make any profit so we're done with this opportunity
            break
    #
    # If we were able to make any profit, then report the opportunity
    if gross > cost:
        for i in all_sell_orders.keys():
            all_sell_orders[i]=compress_order_list(all_sell_orders[i], False)
        return dict(gross=gross, cost=cost, profit=gross - cost, margin=(gross - cost)/cost, 
                    buy_orders=compress_order_list(all_buy_orders), 
                    sell_orders=all_sell_orders)
    return None

# Finally, we can write the complete opportunity finder function
def find_opportunities(order_book, type_map, station_id, region_id, efficiency, sales_tax, station_tax, verbose=False):
    total_snapshots = len(order_book.groupby(order_book.index))
    if verbose:
        print("Checking %d snapshots for opportunities" % total_snapshots)
    opportunities = []
    count = 0
    for snapshot_group in order_book.groupby(order_book.index):
        #
        # Each group is a pair (snapshot_time, snapshot_dataframe)
        snapshot_time = snapshot_group[0]
        snapshot = snapshot_group[1]
        if verbose:
            print("X", end='')
            count += 1
            if count % 72 == 0:
                print()
        #
        # Iterate through each source type looking for opportunities
        for source_type in type_map.values():
            opp = attempt_opportunity(snapshot, source_type['typeID'], region_id, station_id, type_map, 
                                      sales_tax, efficiency, station_tax)
            if opp is not None:
                #
                # Save the time and type if we've found a valid opportunity
                opp['time'] = snapshot_time
                opp['type'] = source_type['typeName']
                opportunities.append(opp)
    if verbose:
        print()
    return opportunities

#
# NOTE: since this cell attempts to capture complete opportunities, execution will take much longer
# than our earlier cell since that cell only attempted to discover whether an opportunity existed.
#
full_opportunities = find_opportunities(order_book, source_types, station_id, region_id, 
                                        efficiency, tax_rate, station_tax, verbose=True)

# We should have the same number of opportunities as before, but the profit should indicate the
# maximum value achievable from each opportunity.  Here are the results.
display_opportunities(full_opportunities)

# Some of the opportunities for this day look very promising.  In preparation for a backtest over
# a longer date range, we'd like to determine the total value of all the opportunities for a
# given day.  Notice, however, that the same opportunity is often available over many consecutive 
# snapshots.  We'd like to count each distinct opportunity exactly once to avoid skewing our results 
# with double counting.  A simple way to do this is to collapse all opportunities for the same type 
# over consecutive snaphots into a single opportunity.  For simplicity, we'll use the first appearance 
# of an opportunity as the representative for consecutive instances of the same opportunity.
# This isn't perfect, but is a reasonable approximation that will make it easier to estimate the
# total opportunity value of a given day.
def clean_opportunities(opps):
    new_opps = []
    stamp_map = {}
    types = set([x['type'] for x in opps])
    # Flatten opportunites for each type
    for next_type in types:
        stamp_list = []
        last = None
        for i in [x['time'] for x in opps if x['type'] == next_type]:
            if last is None:
                # First opportunity
                stamp_list.append(i)
            elif i - last > datetime.timedelta(minutes=5):
                # Start of new run
                stamp_list.append(i)
            last = i
        stamp_map[next_type] = stamp_list
    # Rebuild opportunities by only selecting opportunities in
    # the flattened lists.
    for opp in opps:
        type = opp['type']
        if opp['time'] in stamp_map[type]:
            new_opps.append(opp)
    # Return the new opportunity list
    return new_opps

cleaned_full_opps = clean_opportunities(full_opportunities)
display_opportunities(cleaned_full_opps)

# Finally, we can summarize the performance of our sample day
total_profit = np.sum([x['profit'] for x in cleaned_full_opps])
total_cost = np.sum([x['cost'] for x in cleaned_full_opps])
total_return = total_profit / total_cost
print("Total opportunity profit: %s ISK" % "{:,.2f}".format(total_profit))
print("Total opportunity retrun: %s%% ISK" % "{:,.2f}".format(total_return * 100))

