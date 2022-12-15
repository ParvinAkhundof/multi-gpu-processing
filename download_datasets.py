import requests


url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
r = requests.get(url, allow_redirects=True)

open('train_32x32.mat', 'wb').write(r.content)


url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
r = requests.get(url, allow_redirects=True)

open('test_32x32.mat', 'wb').write(r.content)