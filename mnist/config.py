ip='10.42.0.1'

tf_config={
    'cluster': {'worker': [ip+':11000',ip+':22']},'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"




