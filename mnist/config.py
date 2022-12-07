ip='10.42.0.1'

tf_config={
    'cluster': {'worker': [ip+':22','10.42.0.125:11000']},'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"




