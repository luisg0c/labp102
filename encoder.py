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
