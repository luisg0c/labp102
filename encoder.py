import numpy as np
import pandas as pd

# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

D_MODEL = 64
D_FF = 256
N_CAMADAS = 6
