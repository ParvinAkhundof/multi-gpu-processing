ip='10.42.0.1'

tf_config={
    'cluster': {'worker': [ip+':12345','10.42.0.125:12345']},'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
