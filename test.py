import json

f = open('NewFile.txt', 'w') 
f.write('{"cluster": {"worker": ["192.168.75.40:12345","192.168.75.41:12345","192.168.75.42:12345","192.168.75.43:12345"]},"task": {"type": "worker", "index": 0} }')



f = open("NewFile.txt", "r")
obj=json.loads(f.read())
print(obj['cluster'])


