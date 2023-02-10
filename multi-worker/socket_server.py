import socket
import worker
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
        'worker': ['192.168.75.40:12345','192.168.75.41:12345','192.168.75.42:12345','192.168.75.43:12345']},
    'task': {'type': 'worker', 'index': 0}
}
# ######


import socket
import os
from threading import Thread
import threading

clients = set()
clients_lock = threading.Lock()
ip_list=[]
ip_list.append(my_ip+":12345")

try:
    def listener(client, address):
        print("Accepted connection from: ", address)
        with clients_lock:
            clients.add(client)
        try:
            data=""    
            while data != b'start':
                data = client.recv(1024)
                
                if not data:
                    break
                else:
                    print(repr(data))
                    if(data != b'start'):
                        ip_list.append(data.decode('ascii')+":12345")
                        with clients_lock:
                            for c in clients:
                                c.sendall(str.encode(','.join(ip_list)))
                    else:
                        
                        with clients_lock:
                            for c in clients:
                                c.sendall(data)
            
            tf_config['cluster']['worker']=ip_list
            print(tf_config)
            f = open('tf_config.txt', 'w') 
            f.write(''+json.dumps(tf_config))
            # x=1/0
            worker.run_worker(my_ip)
            
                            
                        
                    
                    
        finally:
            with clients_lock:
                clients.remove(client)
                client.close()
                

    host = my_ip
    port = 5000


    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host,port))
    s.listen(125)
    th = []

    while True:
        print("Server is listening for connections...")
        client, address = s.accept()

        th.append(Thread(target=listener, args = (client,address)).start())

    
except:
    worker.run_worker(my_ip)

