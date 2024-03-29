import socket
import json

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
print("ip of the server: "+my_ip)


tf_config={
    'cluster': {
        'worker': []},
    'task': {'type': 'worker', 'index': 0}
}
# ######


import socket
from threading import Thread
import threading

clients = set()
clients_lock = threading.Lock()
ip_list=[]


def listener(client,address):
    
    with clients_lock:
        clients.add(client)
    try:
        data=""    
        while data != b'start':
            data = client.recv(1024)
            
            if not data:
                break
            else:
                if(data != b'start'):
                    ip_list.append(data.decode('ascii')+":12345")
                    with clients_lock:
                        for c in clients:
                            c.sendall(str.encode(','.join(ip_list)))
                else:
                    tf_config['cluster']['worker']=ip_list
                    f = open('tf_config.txt', 'w') 
                    f.write(''+json.dumps(tf_config))
                    with clients_lock:
                        for c in clients:
                            c.sendall(data)
        
        
        print(tf_config)


        
                        
                    
                
                
    finally:
        with clients_lock:
            clients.remove(client)
            client.close()
            

host = my_ip
port = 5015


s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host,port))
s.listen(125)
th = []


try:
    while True:
        print("Server is listening for connections...")
        client, address = s.accept()
        print("Accepted connection from: ", address)
        th.append(Thread(target=listener, args = (client,address)).start())
except KeyboardInterrupt:
    print('interrupted!')
    s.close()
    




