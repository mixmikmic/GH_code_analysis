import splinter

browser = splinter.Browser()

get_ipython().system('echo $PATH')

get_ipython().system('ls /Users/hupili/Desktop/COMM7780-JOUR7280/python-for-data-and-media-communication/venv/bin')

get_ipython().system('ls /Users/hupili/Desktop/COMM7780-JOUR7280/python-for-data-and-media-communication/venv/bin/geckodriver*')

browser = splinter.Browser()

browser.visit('https://google.com')

browser.url

browser.html[:200]

