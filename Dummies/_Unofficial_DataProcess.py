from matplotlib import pyplot as plt

from transformers import ElectraTokenizer, ElectraForMaskedLM, BertForSequenceClassification
import torch
from torchvision import models

a = torch.randint(low=0, high=100, size=(2, 10))
indices = torch.argmax(a, dim=1)
print(a)
print(indices)
