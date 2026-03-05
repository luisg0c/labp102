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
    x = x - np.max(x, axis=-1, keepdims=True)
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


def encoder(x, n_camadas):
    for i in range(n_camadas):
        Wq = np.random.randn(D_MODEL, D_MODEL)
        Wk = np.random.randn(D_MODEL, D_MODEL)
        Wv = np.random.randn(D_MODEL, D_MODEL)

        W1 = np.random.randn(D_MODEL, D_FF)
        b1 = np.zeros(D_FF)
        W2 = np.random.randn(D_FF, D_MODEL)
        b2 = np.zeros(D_MODEL)

        att_out = attention(x, Wq, Wk, Wv)
        x1 = layer_norm(x + att_out)

        ff_out = feed_forward(x1, W1, b1, W2, b2)
        x = layer_norm(x1 + ff_out)

        print(f"camada {i+1}: {x.shape}")

    return x


if __name__ == "__main__":
    print("frase:", frase)
    print("ids:", ids)
    print("shape entrada:", X.shape)
    print()

    Z = encoder(X, N_CAMADAS)
    print()
    print("shape saida:", Z.shape)
