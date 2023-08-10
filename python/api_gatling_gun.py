import requests

def api_gatling_gun(a, b):
    try:
        requests.post('https://mysite.com/login', params={
            'user':a,
            'password':b
        })
    except Exception:
        # all exceptions here will be issues form requests.post
        # look at your server logs to see what breaks
        pass
    
fuzz(api_gatling_gun)



