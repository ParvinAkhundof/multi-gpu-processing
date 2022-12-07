tf_config={
    'cluster': {
        'worker': ['127.0.0.1:12345', '127.0.0.1:12345']
    },
    'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
