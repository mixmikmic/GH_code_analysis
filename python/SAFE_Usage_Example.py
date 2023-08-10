import SAFE_Auth_Concept as SAFE



auth=SAFE.Auth()

auth.create_acc(b'myTestCredentials',b'myTestCredentials',b'fakeInvite')

auth.create_acc(b'myTestCredentials',b'myTestCredentials',b'fakeInvite')

auth.login(b'myTestCredentials_notCorrect',b'myTestCredentials')

auth.login(b'myTestCredentials',b'myTestCredentials')





auth.login(b'myTestCredentials_notCorrect',b'myTestCredentials')



