tf_config={
    'cluster': {
        'worker': ['10.42.0.1:12345', '10.42.0.1:12345']
    },
    'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
