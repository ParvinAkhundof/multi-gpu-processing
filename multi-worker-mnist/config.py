tf_config={
    'cluster': {
        'worker': ['192.168.2.159:12345', '192.168.2.159:12345']
    },
    'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
