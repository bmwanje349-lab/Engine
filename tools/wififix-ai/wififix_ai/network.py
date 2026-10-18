"""Small two-head MLP using backpropagation implemented in this repository."""
from pathlib import Path
import numpy as np
from .schema import N_FEATURES, CAUSES, ACTIONS

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

def onehot(y, size):
    out = np.zeros((len(y), size), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

class WiFiFixBrain:
    def __init__(self, seed=0, hidden=(128, 64)):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        self.W1 = rng.normal(0, np.sqrt(2/N_FEATURES), (N_FEATURES, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/h1), (h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.Wc = rng.normal(0, np.sqrt(1/h2), (h2, len(CAUSES))).astype(np.float32)
        self.bc = np.zeros(len(CAUSES), dtype=np.float32)
        self.Wa = rng.normal(0, np.sqrt(1/h2), (h2, len(ACTIONS))).astype(np.float32)
        self.ba = np.zeros(len(ACTIONS), dtype=np.float32)

    def forward(self, X, cache=False):
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

    def fit(self, X, y_cause, y_action, epochs=20, batch_size=256, learning_rate=.0015, seed=99):
        X = np.asarray(X, dtype=np.float32)
        y_cause = np.asarray(y_cause, dtype=np.int64)
        y_action = np.asarray(y_action, dtype=np.int64)
        n = len(X)
        if n < 32 or len(y_cause) != n or len(y_action) != n:
            raise ValueError("training arrays are inconsistent or too small")
        rng = np.random.default_rng(seed)
        names = ("W1", "b1", "W2", "b2", "Wc", "bc", "Wa", "ba")
        params = {k: getattr(self, k) for k in names}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        Yc, Ya = onehot(y_cause, len(CAUSES)), onehot(y_action, len(ACTIONS))
        step, history = 0, []
        for _ in range(epochs):
            order = rng.permutation(n)
            batch_losses = []
            for start in range(0, n, batch_size):
                ix = order[start:start+batch_size]
                x, yc, ya = X[ix], Yc[ix], Ya[ix]
                lc, la, cache = self.forward(x, cache=True)
                _, z1, h1, z2, h2 = cache
                pc, pa = softmax(lc), softmax(la)
                bs = len(ix)
                loss = -np.mean(np.sum(yc*np.log(pc+1e-7), axis=1)) - .75*np.mean(np.sum(ya*np.log(pa+1e-7), axis=1))
                batch_losses.append(float(loss))
                dc, da = (pc-yc)/bs, .75*(pa-ya)/bs
                g = {}
                g["Wc"], g["bc"] = h2.T@dc, dc.sum(axis=0)
                g["Wa"], g["ba"] = h2.T@da, da.sum(axis=0)
                dz2 = (dc@self.Wc.T + da@self.Wa.T) * (z2 > 0)
                g["W2"], g["b2"] = h1.T@dz2, dz2.sum(axis=0)
                dz1 = (dz2@self.W2.T) * (z1 > 0)
                g["W1"], g["b1"] = x.T@dz1, dz1.sum(axis=0)
                norm = np.sqrt(sum(float(np.sum(q*q)) for q in g.values()))
                if norm > 5:
                    for k in g:
                        g[k] *= 5/norm
                step += 1
                for k in names:
                    m[k] = .9*m[k] + .1*g[k]
                    v[k] = .999*v[k] + .001*g[k]*g[k]
                    mh = m[k]/(1-.9**step)
                    vh = v[k]/(1-.999**step)
                    params[k] -= learning_rate*mh/(np.sqrt(vh)+1e-8)
            history.append(float(np.mean(batch_losses)))
        return history

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{k: getattr(self, k) for k in ("W1", "b1", "W2", "b2", "Wc", "bc", "Wa", "ba")},
                            schema_version=np.asarray([1], dtype=np.int32))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            w1, w2 = z["W1"], z["W2"]
            model = cls(seed=0, hidden=(w1.shape[1], w2.shape[1]))
            for k in ("W1", "b1", "W2", "b2", "Wc", "bc", "Wa", "ba"):
                setattr(model, k, np.asarray(z[k], dtype=np.float32))
        return model
