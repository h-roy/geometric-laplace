import torch.nn.functional as F
import jax.numpy as jnp
from flax import linen as nn


class ConvNet(nn.Module):
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        if len(x.shape) != 4:
            x = jnp.expand_dims(x, 0)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(features=self.output_dim)(x)
