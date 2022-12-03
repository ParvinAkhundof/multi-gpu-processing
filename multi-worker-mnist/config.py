tf_config={
    'cluster': {
        'worker': ['192.168.75.34:12345', '192.168.75.34:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
