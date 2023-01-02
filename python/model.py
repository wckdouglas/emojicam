import flax.linen as nn
import optax
from jax.example_libraries import optimizers, stax


class EmojiClassifier(nn.Module):
    n_target: int

    def setup(self):
        self.conv1 = nn.Sequential([nn.Conv(features=3, kernel_size=(3, 8), strides=2), nn.tanh])

        self.conv2 = nn.Sequential([nn.Conv(features=8, kernel_size=(32, 3), strides=2), nn.tanh])

        self.linear = nn.Dense(features=self.n_target)

    def __call__(self, x):
        h1 = nn.avg_pool(x, window_shape=(2, 2), padding="SAME")
        h2 = self.conv1(h1)
        h3 = nn.avg_pool(h2, window_shape=(2, 2), padding="SAME")
        h4 = self.conv2(h3)
        h5 = self.linear(h4.reshape((len(h4), -1)))
        return h5

    def encode(self, x):
        return self.encoder(x)
