from matplotlib import pyplot as plt

from transformers import ElectraTokenizer, ElectraForMaskedLM, BertForSequenceClassification
import torch
from torchvision import models

model = models.resnet18()
param_groups = []
model_groups = [model.layer1, model.layer2, model.layer3, model.layer4]
lrs = [1e-1 * i for i in range(len(model_groups))]
for idx, lr in enumerate(lrs):
    param_groups += [{"params": model_groups[idx].parameters(), "lr": lrs[idx]}]
optimizer = torch.optim.Adam(param_groups)
for i in range(len(optimizer.param_groups)):
    print(optimizer.param_groups[i]['lr'])