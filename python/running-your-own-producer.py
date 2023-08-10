import pixiedust
pixiedust.installPackage("https://github.com/ibm-watson-data-lab/advo-beta-producer/raw/master/dist/LocalCartKafkaProducer-1.0-SNAPSHOT-uber.jar")

get_ipython().run_cell_magic('scala', '', 'System.setProperty("KAFKA_USER_NAME","**USER**")\nSystem.setProperty("KAFKA_PASSWORD","**PASSWORD**")\nSystem.setProperty("KAFKA_API_KEY","**API_KEY**")\nSystem.setProperty("USE_JAAS", "true")')

get_ipython().run_cell_magic('scala', '', 'import com.ibm.localcart.DataStream\nprintln(DataStream.getInstance());')

get_ipython().run_cell_magic('scala', '', 'import com.ibm.localcart.DataStream\nprintln(DataStream.getInstance().getStats)')

