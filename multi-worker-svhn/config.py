# tf_config={
#     'cluster': {'worker': ['192.168.2.135:12345','192.168.2.140:23456']},'task': {'type': 'worker', 'index': 0}
# }


# tf_config={
#     'cluster': {'worker': ['192.168.75.27:12345','192.168.75.28:23456']},'task': {'type': 'worker', 'index': 0}
# }


# tf_config={
#     'cluster': {'worker': ['192.168.75.27:12345','192.168.75.28:23456','192.168.75.29:34567']},'task': {'type': 'worker', 'index': 0}
# }


tf_config={
    'cluster': {'worker': ['localhost:12345']},'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
