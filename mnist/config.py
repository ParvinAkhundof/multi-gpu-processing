ip='10.42.0.1'

# tf_config={
#     'cluster': {'worker': [ip+':12345','10.42.0.125:23456']},'task': {'type': 'worker', 'index': 0}
# }

# checkpoint_dir = "./ckpt"




tf_config={
    'cluster': {'worker': [ip+':12345',ip+':23456']},'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"