"""Wide, trainable multi-head WiFiFix model; no pretrained weights are used."""
from pathlib import Path
import numpy as np
from .schema import N_FEATURES, CAUSES, ACTIONS

ARCHITECTURE_VERSION = 3
HIDDEN_SIZES = (3072, 1664)
MODEL_NAME = "WiFiFixBrain-v3-multipath"
MIN_MODEL_BYTES = 20_000_000
MIN_PARAMETER_COUNT = 5_200_000
PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "Wc", "bc", "Wa", "ba")


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def onehot(y, size):
    out = np.zeros((len(y), size), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


class WiFiFixBrain:
    """Two shared representation layers and cause/action prediction heads.

    The saved checkpoint contains more than 5.2 million trainable float32
    parameters (about 20.96 MB decimal of actual weights). The architecture
    stays close to the required 20 MB floor without padding or duplicate data.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        h1, h2 = HIDDEN_SIZES
        self.W1 = rng.normal(0, np.sqrt(2 / N_FEATURES), (N_FEATURES, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.Wc = rng.normal(0, np.sqrt(1 / h2), (h2, len(CAUSES))).astype(np.float32)
        self.bc = np.zeros(len(CAUSES), dtype=np.float32)
        self.Wa = rng.normal(0, np.sqrt(1 / h2), (h2, len(ACTIONS))).astype(np.float32)
        self.ba = np.zeros(len(ACTIONS), dtype=np.float32)

    def _params(self):
        return {name: getattr(self, name) for name in PARAMETER_NAMES}

    @property
    def parameter_count(self):
        return int(sum(param.size for param in self._params().values()))

    @property
    def trainable_bytes(self):
        return int(sum(param.nbytes for param in self._params().values()))

    def forward(self, X, cache=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError("input must have shape (batch, feature_count)")
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        cause_logits = h2 @ self.Wc + self.bc
        action_logits = h2 @ self.Wa + self.ba
        return (cause_logits, action_logits, (X, z1, h1, z2, h2)) if cache else (cause_logits, action_logits)

    def probabilities(self, X):
        c, a = self.forward(np.asarray(X, dtype=np.float32))
        return softmax(c), softmax(a)

    def fit(self, X, y_cause, y_action, epochs=8, batch_size=256, learning_rate=.0015, seed=99):
        X = np.asarray(X, dtype=np.float32)
        y_cause = np.asarray(y_cause, dtype=np.int64)
        y_action = np.asarray(y_action, dtype=np.int64)
        n = len(X)
        if n < 32 or len(y_cause) != n or len(y_action) != n:
            raise ValueError("training arrays are inconsistent or too small")
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError("incorrect feature count")
        if not np.isfinite(X).all() or not np.isfinite(y_cause).all() or not np.isfinite(y_action).all():
            raise ValueError("training arrays contain non-finite values")
        if y_cause.min() < 0 or y_cause.max() >= len(CAUSES) or y_action.min() < 0 or y_action.max() >= len(ACTIONS):
            raise ValueError("label outside the configured class range")

        rng = np.random.default_rng(seed)
        params = self._params()
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        Yc = onehot(y_cause, len(CAUSES))
        Ya = onehot(y_action, len(ACTIONS))
        step, history = 0, []
        for _ in range(epochs):
            order = rng.permutation(n)
            batch_losses = []
            for start in range(0, n, batch_size):
                ix = order[start:start + batch_size]
                x, yc, ya = X[ix], Yc[ix], Ya[ix]
                lc, la, cache = self.forward(x, cache=True)
                _, z1, h1, z2, h2 = cache
                pc, pa = softmax(lc), softmax(la)
                bs = len(ix)
                loss = -np.mean(np.sum(yc * np.log(pc + 1e-7), axis=1))
                loss -= .75 * np.mean(np.sum(ya * np.log(pa + 1e-7), axis=1))
                batch_losses.append(float(loss))
                dc, da = (pc - yc) / bs, .75 * (pa - ya) / bs
                g = {}
                g["Wc"], g["bc"] = h2.T @ dc, dc.sum(axis=0)
                g["Wa"], g["ba"] = h2.T @ da, da.sum(axis=0)
                dz2 = (dc @ self.Wc.T + da @ self.Wa.T) * (z2 > 0)
                g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(axis=0)
                dz1 = (dz2 @ self.W2.T) * (z1 > 0)
                g["W1"], g["b1"] = x.T @ dz1, dz1.sum(axis=0)
                norm = np.sqrt(sum(float(np.sum(q * q)) for q in g.values()))
                if norm > 5:
                    for k in g:
                        g[k] *= 5 / norm
                step += 1
                for k in PARAMETER_NAMES:
                    m[k] = .9 * m[k] + .1 * g[k]
                    v[k] = .999 * v[k] + .001 * g[k] * g[k]
                    mh = m[k] / (1 - .9 ** step)
                    vh = v[k] / (1 - .999 ** step)
                    params[k] -= learning_rate * mh / (np.sqrt(vh) + 1e-8)
            history.append(float(np.mean(batch_losses)))
        return history

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.parameter_count < MIN_PARAMETER_COUNT or self.trainable_bytes < MIN_MODEL_BYTES:
            raise ValueError("model has too few trainable parameters for the configured minimum")
        payload = self._params()
        payload["architecture_version"] = np.asarray([ARCHITECTURE_VERSION], dtype=np.int32)
        payload["hidden_sizes"] = np.asarray(HIDDEN_SIZES, dtype=np.int32)
        # Uncompressed NPZ: the >20 MB floor is made of trained parameters,
        # not arbitrary padding or duplicated data.
        with p.open("wb") as stream:
            np.savez(stream, **payload)
        if p.stat().st_size < MIN_MODEL_BYTES:
            raise ValueError("saved model is below the required 20 MB minimum")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if "architecture_version" not in z or int(z["architecture_version"][0]) != ARCHITECTURE_VERSION:
                raise ValueError("checkpoint architecture is missing or outdated; retrain with WiFiFixBrain v2")
            if "hidden_sizes" not in z or tuple(int(v) for v in z["hidden_sizes"]) != HIDDEN_SIZES:
                raise ValueError("checkpoint hidden dimensions do not match the current model")
            model = cls(seed=0)
            expected = {
                "W1": (N_FEATURES, HIDDEN_SIZES[0]),
                "b1": (HIDDEN_SIZES[0],),
                "W2": (HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
                "b2": (HIDDEN_SIZES[1],),
                "Wc": (HIDDEN_SIZES[1], len(CAUSES)),
                "bc": (len(CAUSES),),
                "Wa": (HIDDEN_SIZES[1], len(ACTIONS)),
                "ba": (len(ACTIONS),),
            }
            arrays = {}
            for name, shape in expected.items():
                if name not in z or z[name].shape != shape:
                    raise ValueError("checkpoint tensor shape mismatch: " + name)
                arr = np.asarray(z[name], dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise ValueError("checkpoint contains non-finite weights")
                arrays[name] = arr
        for name, arr in arrays.items():
            setattr(model, name, arr)
        return model
