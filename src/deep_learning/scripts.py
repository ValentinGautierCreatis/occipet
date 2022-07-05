#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pathlib

from .variational_auto_encoder import VariationalAutoEncoder

def train_model(model: tf.keras.Model, checkpoint_path: str,
                inputs: np.ndarray, labels: np.ndarray) -> None:
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    checkpoint_path = pathlib.Path(checkpoint_dir) / "cp.ckpt"
    data = np.load(data_path)
    model(data[:1,:,:,:])
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)
    train_model(model, str(checkpoint_path.resolve()), data, data)
