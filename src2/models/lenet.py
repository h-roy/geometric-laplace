import jax.numpy as jnp
from jax import nn as jnn
from flax import linen as nn

class LeNet(nn.Module):
    output_dim: int = 10,
    activation: str = "tanh"

    def act_fun(self, x):
        if self.activation == "tanh":
            return jnn.tanh(x)
        if self.activation == "relu":
            return jnn.relu(x)

    @nn.compact
    def __call__(self, x):
        if len(x.shape) != 4:
            x = jnp.expand_dims(x, 0)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=6, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
        x = self.act_fun(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
        x = self.act_fun(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = self.act_fun(x)
        x = nn.Dense(features=84)(x)
        x = self.act_fun(x)
        x = nn.Dense(features=self.output_dim)(x)

        return x