import dopapy
import dopapy.types as dd
import dopapy.trading as dp
from PIL import Image 
from urllib.request import urlopen
from IPython.display import display

import warnings
warnings.filterwarnings('ignore') #Thats just to ignore warnings that are irrelevant for this type of sample

session = dp.create_session(master_password='password123456',
                            keystore_dir='/data/dopamine/keystore_dir_a/',
                            web_server_host="127.0.0.1",
                            web_server_port=8004,
                            dopmn_budget=10000,
                            ether_budget=int(10E18))

print('DOPA Balance:',session.wallet.dopmn_balance)
print('ETH  Balance:',session.wallet.ether_balance)

counter_quote = dp.get_counter_quote(url="https://127.0.0.1:8003", quote_id=0)

service_descriptor = dd.ServiceDescriptor(dd.ServiceRole.CONSUMER,
                                          output_descriptors=[dd.StringDescriptor()],
                                          input_descriptors=[dd.ImageDescriptor()],
                                          remote_sources=[counter_quote])

my_quote = dp.create_quote_services(session=session,
                                    service_descriptor=service_descriptor,
                                    side=dp.BUY,
                                    reward_descriptor=dp.create_reward_descriptor(max_dopamine_price=int(100)))

print("Loading image...")
imageurl = "http://www.slate.com/content/dam/slate/articles/health_and_science/science/2015/07/150730_SCI_Cecil_lion.jpg.CROP.promo-xlarge2.jpg"
img =Image.open(urlopen(imageurl))
display(img)

ai_result = my_quote.get_service().call(dd.Image(img))

ai_result.value

print('DOPA Balance:',session.wallet.dopmn_balance)
print('ETH  Balance:',session.wallet.ether_balance)



