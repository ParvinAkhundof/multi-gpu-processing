from tensorflow import keras
import svhn_setup
import os


def make_or_restore_model(checkpoint_dir):
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return svhn_setup.build_and_compile_cnn_model()