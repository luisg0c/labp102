import numpy as np
from encoder import softmax, attention, layer_norm, feed_forward, encoder, D_MODEL, D_FF


def test_softmax_soma_um():
    x = np.random.randn(1, 4, 10)
    s = softmax(x)
    somas = np.sum(s, axis=-1)
    assert np.allclose(somas, 1.0), f"softmax nao soma 1: {somas}"
    print("ok: softmax soma 1")


def test_attention_mantem_shape():
    x = np.random.randn(1, 5, D_MODEL)
    Wq = np.random.randn(D_MODEL, D_MODEL)
    Wk = np.random.randn(D_MODEL, D_MODEL)
    Wv = np.random.randn(D_MODEL, D_MODEL)
    out = attention(x, Wq, Wk, Wv)
    assert out.shape == x.shape, f"shape errado: {out.shape}"
    print("ok: attention mantem shape")


def test_layer_norm_normaliza():
    x = np.random.randn(1, 5, D_MODEL) * 100 + 50
    normed = layer_norm(x)
    media = np.mean(normed, axis=-1)
    assert np.allclose(media, 0, atol=1e-5), f"media nao eh ~0: {media}"
    print("ok: layer norm normaliza")


def test_encoder_mantem_shape():
    np.random.seed(0)
    x = np.random.randn(1, 5, D_MODEL)
    Z = encoder(x, 6)
    assert Z.shape == (1, 5, D_MODEL), f"shape final errado: {Z.shape}"
    print("ok: encoder mantem shape apos 6 camadas")


if __name__ == "__main__":
    test_softmax_soma_um()
    test_attention_mantem_shape()
    test_layer_norm_normaliza()
    test_encoder_mantem_shape()
    print("\ntodos os testes passaram")
