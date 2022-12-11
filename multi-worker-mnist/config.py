# tf_config={
#     'cluster': {'worker': ['192.168.75.25:12345','192.168.75.27:23456']},'task': {'type': 'worker', 'index': 0}
# }
checkpoint_dir = "./ckpt"

tf_config={
    'cluster': {'worker': ['localhost:12345']},'task': {'type': 'worker', 'index': 0}
}



