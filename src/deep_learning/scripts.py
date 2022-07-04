#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pathlib

from .variational_auto_encoder import VariationalAutoEncoder

def train_model(model: tf.keras.Model, checkpoint_dir: str,
                inputs: np.ndarray, labels: np.ndarray) -> None:
    checkpoint_path = pathlib.Path(checkpoint_dir) / "cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        # save_weights_only=True,
        # verbose=1
    )

    model.fit(
        inputs,
        labels,
        epochs=10,
        callbacks=[cp_callback]
    )



def train_vae(checkpoint_dir: str,
              data_path: str) -> None:

    model = VariationalAutoEncoder((256, 256, 2))
    checkpoint_path = pathlib.Path(checkpoint_dir)
    data = np.load(data_path)
    if checkpoint_path.exists():
        model.load_weights(checkpoint_dir)
    train_model(model, checkpoint_dir, data, data)
