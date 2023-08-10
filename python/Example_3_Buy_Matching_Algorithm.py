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

# Once again, we'll use the most popular station in Jita, Forge for testing
# and we'll imagine we're attempting to sell Tritanium.  So let's retrieve various
# static information.  We'll also set a date so we can grab some book data for testing.
#
sde_client = Client.SDE.get()
type_query = "{values: ['Tritanium']}"
region_query = "{values: ['The Forge']}"
station_query = "{values: ['Jita IV - Moon 4 - Caldari Navy Assembly Plant']}"
type_id = sde_client.Inventory.getTypes(typeName=type_query).result()[0][0]['typeID']
region_id = sde_client.Map.getRegions(regionName=region_query).result()[0][0]['regionID']
station_id = sde_client.Station.getStations(stationName=station_query).result()[0][0]['stationID']
compute_date = convert_raw_time(1483228800000) # 2017-01-01 12:00 AM UTC
print("Using type_id=%d, region_id=%d, station_id=%d at %s" % (type_id, region_id, station_id, str(compute_date)))

# For this simple example, we'll just grab a book snapshot directly from the Orbital Enterprises market data service
# 
mdc_client = Client.MarketData.get()
sample_book = mdc_client.MarketData.book(typeID=type_id, regionID=region_id, date=str(compute_date) + " UTC").result()[0]

# Here's the basic buy matching algorithm with two functions we need to implement
#
def order_match(sell_station_id, buy_station_id, order_range):
    """
    Returns true if a sell market order placed at sell_station_id could be matched
    by a buy order at buy_station_id with the given order_range
    """
    # Case 1 - "region"
    if order_range == 'region':
        return True
    # Case 2 - "station"
    if order_range == 'station':
        return sell_station_id == buy_station_id
    # Remaining checks require solar system IDs and distance between solar systems
    sell_solar = get_solar_system_id(sell_station_id)
    buy_solar = get_solar_system_id(buy_station_id)
    # Make sure we actually found solar systems before continuing.
    # We'll return False if we can't find both solar systems.
    if sell_solar is None or buy_solar is None:
        if sell_solar is None:
            print("Missing solar system for sell station: %d" % sell_station_id)
        if buy_solar is None:
            print("Missing solar system for buy station: %d" % buy_station_id)
        return False
    # 
    # Case 3 - "solarsystem"
    if order_range == 'solarsystem':
        return sell_solar == buy_solar
    # Case 4 - check jump range between solar systems
    jump_count = compute_jumps(sell_solar, buy_solar)
    return jump_count <= int(order_range)

# Before we can use our order matcher, we need to implement the get_solar_system_id and compute_jumps functions.
# For now, we'll assume both stations are not player-owned structures.  This makes the get_solar_system_id
# function a simple SDE lookup.
#
def get_solar_system_id(station_id):
    client = Client.SDE.get()
    station_query = "{values: [" + str(station_id) + "]}"
    result = client.Station.getStations(stationID=station_query).result()[0]
    if len(result) > 0:
        return result[0]['solarSystemID']
    return None

get_solar_system_id(station_id)

# Computing minimum jumps between solar systems is more complicated.
# One way to perform this computation is to apply a bit of graph theory
# and compute a spanning tree over the graph of all solar systems and their
# links via jump gates.  We'll walk through how to do this here.

# First we need some useful functions from the scipy package.
# If you haven't already, you'll need to import these into your
# python install.
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# We'll start by collecting all the solar systems in our target region.
# We'll store the list of solar systems in a dictionary so we can start building
# connectivity between solar systems.
solar_list = sde_client.Map.getSolarSystems(regionID="{values:[" + str(region_id) + "]}").result()[0]
solar_list = [x['solarSystemID'] for x in solar_list]

# Keys in this map are the solar system ID, values are the list of neighboring solar system IDs
# in the same region.
solar_map = {}
for next_solar in solar_list:
    solar_map[next_solar] = [next_solar]

# For each solar system, we need to compute its neighbors in the same region.
# The SDE stores this information as well.
for next_solar in solar_list:
    neighbors = sde_client.Map.getSolarSystemJumps(fromRegionID="{values:[" + str(region_id) + "]}",
                                                   toRegionID="{values:[" + str(region_id) + "]}",
                                                   fromSolarSystemID="{values:[" + str(next_solar) + "]}").result()[0]
    for next_neighbor in neighbors:
        neighbor_id = next_neighbor['toSolarSystemID']
        if neighbor_id not in solar_map[next_solar]:
            solar_map[next_solar].append(neighbor_id)

# Now we're ready to build an adjacency matrix based on computed neighbors.
# We start with an array which scipy will turn into an efficient matrix.
solar_count = len(solar_list)
adj_array = []
for i in range(solar_count):
    next_row = []
    source_solar = solar_list[i]
    for j in range(solar_count):
        dest_solar = solar_list[j]
        if dest_solar in solar_map[source_solar]:
            next_row.append(1)
        else:
            next_row.append(0)
    adj_array.append(next_row)

adj_matrix = csr_matrix(adj_array)

# And finally, we can turn this into a shortest path matrix we can reference for our computations.
shortest_matrix = shortest_path(adj_matrix, directed=False, return_predecessors=False, unweighted=True)

shortest_matrix

# With the shortest path matrix now computed, we can implement compute jumps
#
def compute_jumps(source_solar, dest_solar):
    source_index = solar_list.index(source_solar)
    dest_index = solar_list.index(dest_solar)
    return shortest_matrix[source_index][dest_index]

# We'll test our function on two random solar systems, e.g.
#
# solar_list[10] = 30000129 = Unpas
# solar_list[20] = 30000139 = Urlen
#
# The SDE says these two solar systems are adjacent (share a jump gate):
sde_client.Map.getSolarSystemJumps(fromRegionID="{values:[" + str(region_id) + "]}",
                                   toRegionID="{values:[" + str(region_id) + "]}",
                                   fromSolarSystemID="{values:[ 30000129 ]}",
                                   toSolarSystemID="{values:[ 30000139 ]}").result()[0]

# Our new function should agree and print 1
#
compute_jumps(solar_list[10], solar_list[20])

# Let's try out our matcher on the order book we downloaded.
# If you remember from the previous example, the "orders" field contains all the orders in this snapshot.
# The "buy" field tells us which orders are buys.  Let's pull those out into an array.
#
buy_orders = [x for x in sample_book['orders'] if x['buy']]

# Now let's check which buy orders could match a sell order placed at our target station:
for next_order in buy_orders:
    if order_match(station_id, next_order['locationID'], next_order['orderRange']):
        print("Match: order %d in station %d" % (next_order['orderID'], next_order['locationID']))

# Our matcher works but note that we failed to resolve the solar system for some stations.
# These are player-owned structures and are not recorded in the SDE.  Another way to tell
# these are player-owned structures is from the station ID: station IDs greater than
# 1,000,000,000,000 (1 trillion) are generally player-owned structures.
#
# The official way to look up the solar system for a player-owned structure is to use the
# EVE Swagger Interface.  Specifically, the "universe structures" endpoint:
#
# https://esi.tech.ccp.is/latest/#!/Universe/get_universe_structures_structure_id
#
# However, this endpoint requires OAuth authentication which is beyond the scope of this example.
# We'll cover how to use OAuth authenticated endpoints in a later appendix.  For now, we'll
# use a third party service which tracks player-owned structures and provides access without
# requiring authentication.  You can find more detail about this service here:
#
# https://stop.hammerti.me.uk/api/
#
# The EveKit Client module includes an endpoint for accessing this service.  Let's try it out:
#
po_structure_client = Client.Citadel.get()

po_structure_client.Citadel.getCitadel(citadel_id=1021705628874).result()[0]

# Using this new endpoint, we can now improve our our function to look up solar system IDs
#
def get_solar_system_id(station_id):
    client = Client.SDE.get()
    station_query = "{values: [" + str(station_id) + "]}"
    result = client.Station.getStations(stationID=station_query).result()[0]
    if len(result) > 0:
        return result[0]['solarSystemID']
    # Might be a player-owned structure.  Check for that as well
    client = Client.Citadel.get()
    result = client.Citadel.getCitadel(citadel_id=station_id).result()[0]
    if str(station_id) in result:
        return result[str(station_id)]['systemId']
    return None

get_solar_system_id(1021705628874)

# And, finally, we can try our order matcher again:
#
for next_order in buy_orders:
    if order_match(station_id, next_order['locationID'], next_order['orderRange']):
        print("Match: order %d in station %d" % (next_order['orderID'], next_order['locationID']))

# Our new order matcher now properly resolves player-owned structures as well.  As in previous examples, we 
# now turn to EveKit library functions which simplify or eliminate some of the steps above.
#
# You'll often need to run the order matcher when developing your own trading strategies.  Since the order
# matcher will frequently access map data, the EveKit libraries provide a few modules for making this easier.
#
# The first such library is the Region class which loads Region map information.  You can use this class
# as a cache to speed up frequent solar system lookups.
from evekit.map import Region
region_cache = Region.get_region(region_id)

# You can use this class for non-player-owned structure lookups, e.g.
region_cache.station_map[station_id].__dict__

# This class also computes solar system and constellation jump counts
region_cache.solar_system_jump_count(30000129, 30000139)

# Since order matching is such a frequent need, we've turned it into an EveKit function call
#
from evekit.marketdata import TradingUtil

# You can use TradingUtil.check_range without any other setup as follows:
config = dict(use_citadel=True)
for next_order in buy_orders:
    if TradingUtil.check_range(region_id, station_id, next_order['locationID'], next_order['orderRange'], config):
        print("Match: order %d in station %d" % (next_order['orderID'], next_order['locationID']))

