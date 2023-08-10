import swat

conn = swat.CAS('server-name.mycompany.com', 5570, 'username', 'password')

binconn = swat.CAS('server-name.mycompany.com', 5570, protocol='cas')

restconn = swat.CAS('server-name.mycompany.com', 8888, protocol='http')

tbl = conn.loadtable('data/iris.csv', caslib='casuser').casTable

out = tbl.summary()
out

def result_cb(key, value, response, connection, userdata):
    print('\n>>> RESULT %s\n' % key)
    print(value)
    return userdata

tbl.groupby('species').summary(resultfunc=result_cb, subset=['min', 'max'])

def response_cb(response, connection, userdata):
    print('\n>>> RESPONSE')
    for k, v in response:
        print('\n>>> RESULT %s\n' % k)
        print(v)
    return userdata

tbl.groupby('species').summary(responsefunc=response_cb, subset=['min', 'max'])

conn1 = swat.CAS()
conn2 = swat.CAS()

tbl1 = conn1.loadtable('data/iris.csv', caslib='casuser').casTable
tbl2 = conn2.loadtable('data/iris.csv', caslib='casuser').casTable

conn1 = tbl1.groupby('species').invoke('summary', subset=['min', 'max'])

for resp in conn1:
    print('\n>>> RESPONSE')
    for k, v in resp:
        print('\n>>> RESULT %s\n' % k)
        print(v)

conn1 = tbl1.groupby('species').invoke('summary', subset=['min', 'max'])
conn2 = tbl2.groupby('species').invoke('topk', topk=1, bottomk=1)

for resp, c in swat.getnext(conn1, conn2):
    print('\n>>> RESPONSE')
    for k, v in resp:
        print('\n>>> RESULT %s\n' % k)
        print(v)

conn

conn.listsessions()

conn2 = swat.CAS('server-name.mycompany.com', 5570, session='2fed25d6-4946-0540-8308-73c7f75b53c6')

conn2



