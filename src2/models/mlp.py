import jax.numpy as jnp
from jax import nn as jnn
from flax import linen as nn

class MLP(nn.Module): 
    output_dim: 1
    hidden_dim: 64
    num_layers: 3
    activation: "tanh"

    def act_fun(self, x):
        if self.activation == "tanh":
            return jnn.tanh(x)
        if self.activation == "relu":
            return jnn.relu(x)
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.act_fun(x)
        x = nn.Dense(self.output_dim)(x) 
        return x