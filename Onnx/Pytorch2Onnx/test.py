import urllib3

default_headers = urllib3.util.make_headers(proxy_basic_auth='user:pwd')
http = urllib3.ProxyManager(proxyUrl, proxy_headers=default_headers, ca_certs=certifi.where())
http.request('GET', url)