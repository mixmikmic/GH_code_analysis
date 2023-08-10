import gridftp_log_info
# for development purpose: automatic reload of code
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

file_path = ("/home/stephan/Repos/DKRZ-gitlab/data-manager-1/Data/esg-server-usage-gridftp.log-full")
# time format =  YYYYMMDDHHMMSS
time_interval = [20160100000000,
                 20181000000000]

volumes, transfer_rates, client_ips, user_openids = gridftp_log_info.gridftplog_to_dict(file_path,time_interval,select="user")

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
exclude_ips = ["136.172.30.85",
               "136.172.18.4",
               "::ffff:136.172.18.11",
               "::ffff:136.172.18.111",
               "136.172.13.20",
               "136.172.13.100",
               "136.172.18.100",
               "136.172.18.190",
               "136.###"]

for client in client_ips:
   if client in volumes.keys() and client not in exclude_ips:
       target = 'unknown'
       if len(client.split(".")) == 4:
          # assume valid ip
          geo_info = gridftp_log_info.get_geolocation_for_ip(client)  
          target = geo_info['city']+" in "+geo_info['country_name']
       print "Transfer to: ",target
       plt.scatter(volumes[client],transfer_rates[client],label='from esgf1.dkrz.de (gridftp)')
       plt.xlabel('Volume (Gbytes)')
       plt.ylabel('Transfer Rate (Mbyte/s)') 
       plt.title("To "+client)
       plt.legend()
       plt.show()
   else:
       if client not in volumes.keys():
          print "No valid log entry for client ip",client
       else:
          print "log entry not shown for client ip",client

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

for user in user_openids:
   print "User: ",user
   plt.scatter(volumes[user],transfer_rates[user],label='from esgf1.dkrz.de (gridftp)')
   plt.xlabel('Volume (Gbytes)')
   plt.ylabel('Transfer Rate (Mbyte/s)') 
   plt.title(user)
   plt.legend()
   plt.show()

# attention: not ready .. just coarse look based on DataFrame visualization
# to be completed ...
import pandas
for user in client_ips:
  try:
        plot_df = pandas.DataFrame({'volume':volumes[user],'transfer speed':transfer_rates[user]})
        plot_df.plot(y='transfer speed',title=user)
  except:
    continue
#plot_df.plot(kind='box')



