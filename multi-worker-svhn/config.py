tf_config={
    'cluster': {
        'worker': ['172.31.80.132:12345', '172.31.80.132:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

checkpoint_dir = "./ckpt"
