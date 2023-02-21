#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
    MaxPool2D,
)


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_3x3(out_dim):
    return Conv2D(
        filters=out_dim, kernel_size=3, strides=1, padding="same", activation="gelu"
    )


def get_1x1(out_dim):
    return Conv2D(
        filters=out_dim, kernel_size=1, strides=1, padding="same", activation="gelu"
    )


class ResBlock(keras.layers.Layer):
    def __init__(
        self,
        middle_depth,
        out_depth,
        down_rate=None,
        residual=False,
        use_3x3=True,
        **kwargs
    ) -> None:
        super(ResBlock, self).__init__(**kwargs)
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(middle_depth)
        self.c2 = get_3x3(middle_depth) if use_3x3 else get_1x1(middle_depth)
        self.c3 = get_3x3(middle_depth) if use_3x3 else get_1x1(middle_depth)
        self.c4 = get_1x1(out_depth)
        self.maxpooling = (
            tf.keras.Sequential([MaxPool2D(pool_size=down_rate)])
            if down_rate is not None
            else tf.keras.Sequential()
        )

    def call(self, x):
        xhat = self.c1(x)
        xhat = self.c2(xhat)
        xhat = self.c3(xhat)
        xhat = self.c4(xhat)
        out = x + xhat if self.residual else xhat
        out = self.maxpooling(out)
        return out


class ConvBlock(keras.layers.Layer):
    # Stride is 2 so divide size by 2 each time
    def __init__(self, nb_filters, **kwargs) -> None:
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(nb_filters, kernel_size=(3, 3), padding="same", strides=2)
        self.normalize = BatchNormalization()
        self.activation = LeakyReLU()

    def call(self, x):
        xhat = self.conv(x)
        # xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat


class DeconvBlock(keras.layers.Layer):
    # Stride is 2 so divide size by 2 each time
    def __init__(self, nb_filters, **kwargs) -> None:
        super(DeconvBlock, self).__init__(**kwargs)
        self.deconv = Conv2DTranspose(
            nb_filters, kernel_size=(3, 3), padding="same", strides=2
        )
        self.normalize = BatchNormalization()
        self.activation = LeakyReLU()

    def call(self, x):
        xhat = self.deconv(x)
        # xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat
