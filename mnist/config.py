# ip='10.42.0.1'

# tf_config={
#     'cluster': {'worker': [ip+':12345','10.42.0.125:23456']},'task': {'type': 'worker', 'index': 0}
# }

checkpoint_dir = "./ckpt"





# tf_config={
#     'cluster': {'worker': ['192.168.2.135:12345','192.168.2.140:23456']},'task': {'type': 'worker', 'index': 0}
# }


tf_config={
    'cluster': {'worker': ['192.168.75.25:12345','192.168.75.27:23456']},'task': {'type': 'worker', 'index': 0}
}