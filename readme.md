agent de lecture d un mail

## Binarisation de GPT-2

Le fichier `binary_gpt2.py` contient un prototype permettant de convertir les couches `nn.Linear` d'un `GPT2LMHeadModel` en versions binarisées. Les poids et les activations sont réduits à ±1 à l'aide d'une fonction `SignSTE` (Straight Through Estimator) pour conserver un entraînement possible.

Exemple d'utilisation :
```python
from transformers import AutoConfig
from binary_gpt2 import GPT2BinaryModel

config = AutoConfig.from_pretrained("distilgpt2")
model = GPT2BinaryModel(config)
# Le modèle peut ensuite être chargé avec des poids pré-entraînés
```
