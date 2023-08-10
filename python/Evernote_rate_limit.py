import evernote

from time import sleep
from evernote.edam.error.ttypes import (EDAMSystemException, EDAMErrorCode)

def evernote_rate_limit(f):
    def f2(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except EDAMSystemException, e:
            if e.errorCode == EDAMErrorCode.RATE_LIMIT_REACHED:
                sleep(e.rateLimitDuration)
                return f(*args, **kwargs)
    
    return f2


class RateLimitingEvernoteProxy(object):
    __slots__ = ["_obj"]
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)
    
    def __getattribute__(self, name):
        return evernote_rate_limit(getattr(object.__getattribute__(self, "_obj"), name))

# settings holds the devToken/authToken that can be used to access Evernote account
#http://dev.evernote.com/doc/articles/authentication.php#devtoken
# settings.authToken

import settings


from evernote.api.client import EvernoteClient

dev_token = settings.authToken

client = RateLimitingEvernoteProxy(EvernoteClient(token=dev_token, sandbox=False))

userStore = client.get_user_store()
user = userStore.getUser()
print user.username

type(client)

