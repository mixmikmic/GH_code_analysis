graphlab_create_product_key = 'YOUR_PRODUCT_KEY'
aws_access_key_id='YOUR_ACCESS_KEY'
aws_secret_access_key='YOUR_SECRET_KEY'

running_on_ec2 = True
# running_on_ec2 = False

get_ipython().run_cell_magic('bash', '', '# initialize filesystem on SSD drives\nsudo mkfs -t ext4 /dev/xvdb\nsudo mkfs -t ext4 /dev/xvdc\n\n# create mount points for SSD drives\nsudo mkdir -p /mnt/tmp1\nsudo mkdir -p /mnt/tmp2\n\n# mount SSD drives on created points and temporary file locations\nsudo mount /dev/xvdb /mnt/tmp1\nsudo mount /dev/xvdc /mnt/tmp2\nsudo mount /dev/xvdb /tmp\nsudo mount /dev/xvdc /var/tmp\n\n# set permissions for mounted locations\nsudo chown ubuntu:ubuntu /mnt/tmp1\nsudo chown ubuntu:ubuntu /mnt/tmp2')

# Fill in YOUR_PRODUCT_KEY which you got from Dato; and from your AWS credentials, YOUR_ACCESS_KEY and YOUR_SECRET_KEY 
import graphlab as gl

if gl.product_key.get_product_key() is None:
    gl.product_key.set_product_key(graphlab_create_product_key)

try:
    gl.aws.get_credentials()
except KeyError:
    gl.aws.set_credentials(access_key_id=aws_access_key_id, 
                       secret_access_key=aws_secret_access_key)

# Set the cache locations to the SSDs.
if running_on_ec2:
    gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", "/mnt/tmp1:/mnt/tmp2")

# Load the CommonCrawl 2012 SGraph
s3_sgraph_path = "s3://dato-datasets-oregon/webgraphs/sgraph/common_crawl_2012_sgraph"
g = gl.load_sgraph(s3_sgraph_path)

# Run PageRank over the SGraph
pr = gl.pagerank.create(g)

# Print results
print "Done! Resulting PageRank model:"
print
print pr

# Print timings
from datetime import timedelta
training_time_secs = pr['training_time']
print "Total training time:", timedelta(seconds=training_time_secs)
print "Avg. time per iteration:", timedelta(seconds=(training_time_secs / float(pr['num_iterations'])))

