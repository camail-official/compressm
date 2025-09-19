import equinox as eqx
import jax
import jax.random as jr


class DualHead(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, in_features, hidden_features, out_features, key):
        key1, key2, key3 = jr.split(key, 3)
        self.layer1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.layer2 = eqx.nn.Linear(hidden_features, int(hidden_features // 2), key=key2)
        self.layer3 = eqx.nn.Linear(int(hidden_features // 2), out_features, key=key3)

    def __call__(self, x):
        x = self.layer1(x)
        x = jax.nn.relu(x)
        x = self.layer2(x)
        x = jax.nn.relu(x)
        x = self.layer3(x)
        return x