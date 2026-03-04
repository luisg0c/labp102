import numpy as np
import pandas as pd

# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

D_MODEL = 64
D_FF = 256
N_CAMADAS = 6

np.random.seed(7)

vocab = pd.DataFrame({
    "palavra": ["cada", "neural", "a", "frase", "rede", "processa", "da", "palavra"],
    "id": range(8)
})

frase = "a rede neural processa cada palavra da frase"
tokens = frase.split()
ids = [int(vocab[vocab["palavra"] == t]["id"].values[0]) for t in tokens]

embedding_table = np.random.randn(len(vocab), D_MODEL)
X = embedding_table[ids]
X = X.reshape(1, len(ids), D_MODEL)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(x, Wq, Wk, Wv):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V


def layer_norm(x, eps=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(var + eps)


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2
