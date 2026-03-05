# Transformer Encoder from Scratch

Implementacao do forward pass de um bloco Encoder do Transformer (Vaswani et al., 2017) usando so NumPy e Pandas.

O encoder recebe uma frase, converte em embeddings e passa por 6 camadas identicas com self-attention, layer norm e feed-forward network.

## Como rodar

Precisa de Python 3, numpy e pandas.

```
pip install numpy pandas
python encoder.py
```

Pra rodar os testes:

```
python test_encoder.py
```

## Estrutura

- `encoder.py` — codigo principal com todas as funcoes e o encoder
- `test_encoder.py` — testes basicos de shape e normalizacao
