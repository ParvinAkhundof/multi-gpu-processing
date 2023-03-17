import socket
import time
import worker

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

my_ip=get_ip()

############

import sys


socketObject = socket.socket()


cluster=""+my_ip.split('.')[0]+"."+my_ip.split('.')[1]+"."+my_ip.split('.')[2]+"."

ip=cluster+sys.argv[1]
try:
    socketObject.connect((ip, 5015))
    print("Connected to "+ip)
except:
    print("not Connected to "+ip)



HTTPMessage = my_ip
bytes = str.encode(HTTPMessage)
socketObject.sendall(bytes)


# Receive the data
iplist=""
run=True
while (True):
    time.sleep(1)

    data = socketObject.recv(1024)
    

    print(data)

    if(len(sys.argv)>2 and run):
        run=False
        HTTPMessage=sys.argv[2]
        bytes = str.encode(HTTPMessage)
        socketObject.sendall(bytes)

    if (data == b'start'):

        print("Connection closed")

        break
    else:
        iplist=data.decode('ascii')


socketObject.close()

tf_config={
    'cluster': {
        'worker': ['192.168.75.40:12345','192.168.75.41:12345','192.168.75.42:12345','192.168.75.43:12345']},
    'task': {'type': 'worker', 'index': 0}
}
tf_config['cluster']['worker']=iplist.split(",")

worker.run_worker(my_ip,tf_config)
