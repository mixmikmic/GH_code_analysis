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

# Load ore and ice types as well as the materials they can refine to.
# Start with ore and ice material groups.
#
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

# Set up types for which we need market data.
#
download_types = set(source_types.keys())
for next_type in source_types.values():
    download_types = download_types.union(next_type['material_map'].keys())

# We assume you've already downloaded market data for our sample day.  If you haven't done this, you
# can retrieve market data by executing the appropriate cell in  Example 7.  We recommend you always 
# download order book data, but if you'd rather use online data, you can remove the "local_storage" 
# argument from the order book functions below.  This cell loads order book data for our target day.
#
from evekit.marketdata import OrderBook
order_book = OrderBook.get_data_frame(dates=[compute_date], types=download_types, regions=[region_id], 
                                      config=dict(local_storage=".", tree=True, skip_missing=True, 
                                                  fill_gaps=True, verbose=True))

# Using limit orders instead of market orders when selling materials requires that we control the
# size of our limit orders.  Otherwise, we'll simply flood the market with excessive limit orders
# which are unlikely to be filled.  We control limit orders by constraining them to be no larger
# than a fraction of historic volume.  As a result, we need historic volume for our target day.
# If this were a live trading strategy, we'd use recent historic volume (e.g. a weighted average
# of recent days) since data for the current day is likely not available.
#
from evekit.marketdata import MarketHistory
market_history = MarketHistory.get_data_frame(dates=[compute_date], types=download_types, regions=[region_id], 
                                              config=dict(local_storage=".", tree=True, skip_missing=True, verbose=True))

# We need two new configuration parameters when using limit orders:
#
# 1. broker_rate - this is the percentage fee charged to place market orders
# 2. volume_limit - this is the fraction of daily volume we will not exceed for limit orders.
#
# In addition to setting efficiency,tax rate and station tax, we set these
# additional parameters at the end of this cell.

# This is the efficiency at a typical NPC station with max skills
efficiency = 0.5 * 1.15 * 1.1 * 1.1

# This is the sales tax at a typical NPC station with max skills
tax_rate = 0.01

# Station tax can be no greater than 0.05.  This value is zero if standings are 6.67 or better.
# As noted above, we're substituting order price for adjusted price.  From empirical observation,
# setting station_tax to 0.04 roughly approximates a station_tax of 0.05 with true adjusted prices.
# So we'll set station tax to 0.04 here.
station_tax = 0.04

# This is the broker rate at a typical NPC station with max skills
broker_rate = 0.025

# A rule of thumb is that we shouldn't be attempting to sell more than 10% of daily
# volume for a partcular asset type.  Since we may be selling across multiple opportunities
# in a given day, we reduce this limit even further to 1% per opportunity.
volume_limit = 0.01

# Dump opportunities in a nice format.  Note that this dumper also displays whether
# any orders were sold as limit orders instead of market orders (see below).
def display_opportunities(opps):
    for opp in opps:
        profit = "{:15,.2f}".format(opp['profit'])
        margin = "{:8.2f}".format(opp['margin'] * 100)
        limit = True
        print("ArbOpp time=%s  profit=%s  return=%s%%  limit=%s  type=%s" % (str(opp['time']), profit, margin, 
                                                                             limit, opp['type']))
    print("Total opportunities: %d" % len(opps))

# We'll move right to optimizing for each opportunity and use the list-based order functions from Example 7.

# Attempt to buy from a list of orders which are assumed to already be filtered to sell orders 
# of the given type and the appropriate location.  This function will consume orders to fill 
# the given volume, and will return a list of objects {price, volume} showing the orders that 
# were made.  This list will be empty if we can not fill the order completely.
def attempt_buy_type_list(buy_volume, sell_order_list):
    potential = []
    for next_order in sell_order_list:
        if buy_volume >= next_order['min_volume'] and next_order['volume'] > 0:
            # Buy into this order
            amount = min(buy_volume, next_order['volume'])
            order_record = dict(price=next_order['price'], volume=amount, market=True)
            buy_volume -= amount
            next_order['volume'] -= amount
            potential.append(order_record)
        if buy_volume == 0:
            # We've completely filled this order
            return potential
    # If we never completely fill the order then return no orders
    return []

# Attempt to sell to a list of orders which are assumed to already be filtered to buy 
# orders of the given type.  We use our range checker to implement proper ranged buy 
# order matching.  This function will consume volume from an order if possible, and 
# return a list of objects {price, volume} showing the orders that were filled.  This 
# list will be empty if we can not fill the order completely.
from evekit.marketdata import TradingUtil

def attempt_sell_type_list(sell_region_id, sell_location_id, sell_volume, buy_order_list):
    config = dict(use_citadel=False)
    potential = []
    for next_order in buy_order_list:
        try:
            if sell_volume >= next_order['min_volume'] and next_order['volume'] > 0 and                TradingUtil.check_range(sell_region_id, sell_location_id, next_order['location_id'], 
                                       next_order['order_range'], config):
                # Sell into this order
                amount = min(sell_volume, next_order['volume'])
                order_record = dict(price=next_order['price'], volume=amount, market=True)
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
    market = True
    for next_order in order_list:
        if next_order['price'] not in order_map:
            order_map[next_order['price']] = next_order['volume']
        else:
            order_map[next_order['price']] += next_order['volume']
        market = market and next_order['market']
    orders = [ dict(price=k,volume=v,market=market) for k, v in order_map.items()]
    return sorted(orders, key=lambda x: x['price'], reverse=not ascending)

# To determine whether we should be selling with limit orders, we need to 
# compute the "spread return" for a given asset.  The following function 
# does this, returning the following structure:
#
# {
#   spread_return: spread return value, or 0 if none
#   best_bid: best bid price, or None
#   best_ask: best ask price, or None
# }
#
def spread_return(snapshot, type_id, station_id, region_id):
    # Attempt to compute best ask
    sell_orders = extract_sell_orders(snapshot, type_id, station_id)
    if len(sell_orders) == 0:
        return dict(spread_return=0, best_bid=None, best_ask=None)
    best_ask = sell_orders[0]['price']
    # Attempt to compute best bid
    buy_orders = extract_buy_orders(snapshot, type_id)
    config = dict(use_citadel=False)
    best_bid = None
    for next_order in buy_orders:
        try:
            if TradingUtil.check_range(region_id, station_id, next_order['location_id'], next_order['order_range'], config):
                best_bid = next_order['price']
                break
        except:
            # We'll get an exception if TradingUtil can't find the location of a player-owned
            # station.  We'll ignore those for now.  Change "use_citadeL" to True above
            # if you'd like to attempt to resolve the location of these stations from a 
            # third party source.
            pass
    if best_bid is None:
        return dict(spread_return=0, best_bid=None, best_ask=None)
    # Return ratio data
    return dict(spread_return=(best_ask - best_bid) / best_bid, best_bid=best_bid, best_ask=best_ask)

# This is our modified opportunity attempter.  We now include the broker fee and determine whether
# it is more profitable to sell with limit or buy orders.  As before, the result of this function 
# will be None if no opportunity was available, or an object:
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
# Compressed order lists group orders by price and sum the volume.  Each order will now include
# the field "market" which is true if the order was a market order, and false otherwise.
#
def attempt_opportunity(snapshot, type_id, region_id, station_id, type_map, tax_rate, efficiency, 
                        station_tax, broker_fee, market_summary, volume_limit):
    # Compute limit order return threshold
    limit_threshold = broker_fee / (1 - tax_rate - broker_fee)
    # Reduce to type to extract minimum reprocessing volume
    by_type = snapshot[snapshot.type_id == type_id]
    required_volume = type_map[type_id]['portionSize']
    #
    # Create source sell order list.
    sell_order_list = extract_sell_orders(snapshot, type_id, station_id)
    #
    # Create refined materials buy order lists and other maps.
    buy_order_map = {}
    all_sell_orders = {}
    limit_order_max = {}
    spread_data = {}
    for next_mat in type_map[type_id]['material_map'].values():
        mat_type_id = next_mat['materialTypeID']
        # Extract the available buy orders for this material
        buy_order_map[mat_type_id] = extract_buy_orders(snapshot, mat_type_id)
        # Track sell orders for this material
        all_sell_orders[mat_type_id] = []
        # Set the total volume limit for sell limit orders for this material
        limit_order_max[mat_type_id] = list(market_summary[market_summary.type_id == mat_type_id]['volume'])[0] * volume_limit
        # Capture spread data for this material
        spread_data[mat_type_id] = spread_return(snapshot, mat_type_id, station_id, region_id)
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
            # We'll use limit orders when selling this material if the spread
            # return exceeds the limit threshold.
            sr_data = spread_data[next_mat_id]
            limit_sell = sr_data['spread_return'] > limit_threshold            
            sell_volume = int(type_map[type_id]['material_map'][next_mat_id]['quantity'] * efficiency)
            # Either sell with limit orders or to the market
            if limit_sell:
                # Selling with limit orders.  Total volume may be limited.
                amount = min(sell_volume, limit_order_max[next_mat_id])
                if amount > 0:
                    sold = [ dict(price=sr_data['best_ask'], volume=amount, market=False) ]
                    limit_order_max[next_mat_id] -= amount
                else:
                    # We can't sell any more volume with limit orders so we're done.  An improvement
                    # would be to now switch to market orders.  We leave this modification to the
                    # reader.
                    sold = []
            else:
                # Selling to the market
                sold = attempt_sell_type_list(region_id, station_id, sell_volume, buy_order_map[next_mat_id])
            if len(sold) == 0:
                # Can't sell any more refined material, done with this opportunity
                sell_orders = []
                break
            #
            # Add gross profit from selling refined material.
            # Include the broker fee if this was a limit sell.
            gross_ratio = (1 - tax_rate - broker_fee) if limit_sell else (1 - tax_rate)
            current_gross += gross_ratio * np.sum([ x['price'] * x['volume'] for x in sold ])
            #
            # Add incremental cost of refining source to this refined material.
            # If we had actual adjusted_prices, we'd use those prices in place of x['price'] below.
            current_cost += station_tax * np.sum([ x['price'] * x['volume'] for x in sold ])
            #
            # Save the set of sell orders we just made
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
def find_opportunities(order_book, type_map, station_id, region_id, efficiency, sales_tax, 
                       station_tax, broker_fee, market_summary, volume_limit, verbose=False):
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
                                      sales_tax, efficiency, station_tax, broker_fee, market_summary,
                                      volume_limit)
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
# NOTE: this cell takes about an hour on our equipment to execute due to attempting to
# capture all opportunities, as well as handling limit sell orders.
#
full_opportunities = find_opportunities(order_book, source_types, station_id, region_id, 
                                        efficiency, tax_rate, station_tax, broker_rate, 
                                        market_history, volume_limit, verbose=True)

# As before, we will "clean" the opportunity list to avoid double counting.
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

# Now let's look at the results including any limit order opportunities.
cleaned_full_opps = clean_opportunities(full_opportunities)
display_opportunities(cleaned_full_opps)

# Aggregate performance for this day with limit orders
total_profit = np.sum([x['profit'] for x in cleaned_full_opps])
total_cost = np.sum([x['cost'] for x in cleaned_full_opps])
total_return = total_profit / total_cost
print("Total opportunity profit: %s ISK" % "{:,.2f}".format(total_profit))
print("Total opportunity retrun: %s%%" % "{:,.2f}".format(total_return * 100))

