#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pathlib
import datetime

from tools.parameters import Parameters
from .variational_auto_encoder import BetaVAE, VariationalAutoEncoder

def train_model(model: tf.keras.Model, checkpoint_path: str,
                inputs: np.ndarray, labels: np.ndarray,
                nb_epochs=100, batch_size=32,
                validation_split=0.0,
                tensorboard=False) -> None:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        # save_weights_only=True,
        # verbose=1
    )

    log_dir = str(pathlib.Path(checkpoint_path) / ".." / "logs" /
                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.Tensorboard(
        log_dir=log_dir
    )

    callbacks = [cp_callback]
    if tensorboard:
        callbacks.append(tensorboard_callback)

    model.fit(
        inputs,
        labels,
        epochs=nb_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=validation_split
    )



def train_vae(checkpoint_dir: str,
              data_path: str, latent_dim: int,
              nb_epochs = 100, batch_size=32,
              learning_rate=1e-3,
              validation_split=0.0,
              tensorboard=False) -> None:

    tf.keras.backend.clear_session()
    model = VariationalAutoEncoder((256, 256, 2), latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)#, loss=tf.keras.losses.MeanSquaredError())
    checkpoint_path = pathlib.Path(checkpoint_dir)
    data = np.load(data_path)
    model(data[:1,:,:,:])
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    train_model(model, str(checkpoint_path.resolve()), data, data,
                nb_epochs, batch_size, validation_split, tensorboard)


def train_vae_param(parameters: Parameters) -> None:
    """
    train_vae(parameters["checkpoint_dir"], parameters["data_path"],
              parameters["latent_dim"], parameters["nb_epochs"],
              parameters["batch_size"], parameters["learning_rate"],
              parameters["validation_split"])
    """
    train_vae(**{k: v for k,v in parameters.data.items()})

def train_beta_vae(checkpoint_dir: str,
                   data_path: str, latent_dim: int, beta: float,
                   nb_epochs = 100, batch_size=32,
                   learning_rate=1e-3,
                   validation_split=0.0,
                   tensorboard=False) -> None:

    tf.keras.backend.clear_session()
    model = BetaVAE((256, 256, 2), latent_dim=latent_dim, beta=beta)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)#, loss=tf.keras.losses.MeanSquaredError())
    checkpoint_path = pathlib.Path(checkpoint_dir)
    data = np.load(data_path)
    model(data[:1,:,:,:])
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    train_model(model, str(checkpoint_path.resolve()), data, data,
                nb_epochs, batch_size, validation_split, tensorboard)


def train_beta_vae_param(parameters: Parameters) -> None:

    train_beta_vae(**{k: v for k,v in parameters.data.items()})
