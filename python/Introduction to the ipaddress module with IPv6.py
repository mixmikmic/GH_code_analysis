import ipaddress
from ipaddress import IPv6Address, IPv6Network, IPv6Interface

ipaddress.ip_address("2001:0db8::beef")

ipaddress.ip_network("2001:0db8:1::/64")

ipaddress.ip_network("2001:0db8:1::affe/64", strict=False)

intf = ipaddress.ip_interface("2001:0db8:1::affe/64")
intf

intf.ip

intf.network

# display the long representation of the IPv6 address
intf.exploded

# display the short version of the IPv6 address
IPv6Address("2001:0db8:0032:0000:beef:0123:cafe:0bd1").compressed

intf = ipaddress.ip_interface("2001:0db8:1::affe/64")
intf.version

intf.with_netmask

intf.with_prefixlen

intf.network.with_netmask

intf.with_hostmask

intf.network.is_private

intf.network.is_reserved

intf.network.is_global

intf.network.is_multicast

intf.network.broadcast_address

intf.network.network_address

intf.network.num_addresses

# be careful with the following function and /64 prefixes because of the large amount of addresses
list(IPv6Network("2001:db8:0:1::/125").hosts())

ipaddr = IPv6Address("2001:db8:0:1::beef")

ipaddr in IPv6Network("2001:db8:0:1::/64")

ipaddr in IPv6Network("2001:db8:0:2::/64")

ipnet = IPv6Network("2001:db8:0:1::/64")

# prefixlen_diff = number of additional network bits
list(ipnet.subnets(prefixlen_diff=4))

# new_prefix = number of network bits for the new prefix
list(ipnet.subnets(new_prefix=68))



