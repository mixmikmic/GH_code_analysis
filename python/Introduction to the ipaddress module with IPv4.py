import ipaddress
from ipaddress import IPv4Address, IPv4Network, IPv4Interface

ipaddress.ip_address("10.10.10.128")

ipaddress.ip_network("10.10.10.0/24")

ipaddress.ip_network("10.10.10.0/255.255.255.0")

ipaddress.ip_network("10.10.10.5/24", strict=False)

intf = ipaddress.ip_interface("10.10.10.5/24")
intf

intf.ip

intf.network

intf = ipaddress.ip_interface("10.10.10.5/24")
intf.version

intf.with_netmask

intf.with_prefixlen

intf.network.with_netmask

intf.with_hostmask

intf.network.is_private

intf.network.is_reserved

intf.network.is_global

intf.ip.is_multicast

intf.network.broadcast_address

intf.network.network_address

intf.network.num_addresses

list(IPv4Network("10.1.1.0/29").hosts())

ipaddr = IPv4Address("192.168.1.23")

ipaddr in IPv4Network("192.168.1.0/24")

ipaddr in IPv4Network("192.168.2.0/24")

ipnet = IPv4Network("10.1.0.0/16")

# prefixlen_diff = number of additional network bits
list(ipnet.subnets(prefixlen_diff=4))

# new_prefix = number of network bits for the new prefix
list(ipnet.subnets(new_prefix=20))



