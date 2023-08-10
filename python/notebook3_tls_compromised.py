from scapy.all import *

record1_str = open('../notebooks/raw_data/tls_session_compromised/01_cli.raw').read()
record1 = TLS(record1_str)
record1.msg[0].show()

record2_str = open('../notebooks/raw_data/tls_session_compromised/02_srv.raw').read()
record2 = TLS(record2_str, tls_session=record1.tls_session.mirror())
record2.msg[0].show()

record3_str = open('../notebooks/raw_data/tls_session_compromised/03_cli.raw').read()
record3 = TLS(record3_str, tls_session=record2.tls_session.mirror())
record3.show()

record4_str = open('../notebooks/raw_data/tls_session_compromised/04_srv.raw').read()
record4 = TLS(record4_str, tls_session=record3.tls_session.mirror())
record4.show()

record5_str = open('../notebooks/raw_data/tls_session_compromised/05_cli.raw').read()
record5 = TLS(record5_str, tls_session=record4.tls_session.mirror())
record5.show()

# Now suppose we possess the private key of the server
# Let's reload the records and register the key to the session
record1_str = open('../notebooks/raw_data/tls_session_compromised/01_cli.raw').read()
record1 = TLS(record1_str)
record2_str = open('../notebooks/raw_data/tls_session_compromised/02_srv.raw').read()
record2 = TLS(record2_str, tls_session=record1.tls_session.mirror())
key = PrivKey('../notebooks/raw_data/pki/srv_key.pem')
record2.tls_session.server_rsa_key = key

record3_str = open('../notebooks/raw_data/tls_session_compromised/03_cli.raw').read()
record3 = TLS(record3_str, tls_session=record2.tls_session.mirror())
record3.show()

record4_str = open('../notebooks/raw_data/tls_session_compromised/04_srv.raw').read()
record4 = TLS(record4_str, tls_session=record3.tls_session.mirror())
record4.show()

record5_str = open('../notebooks/raw_data/tls_session_compromised/05_cli.raw').read()
record5 = TLS(record5_str, tls_session=record4.tls_session.mirror())
record5.show()

