from torchvision import models

model_A = models.resnet18()
model_B = models.vgg19()

params = set(list(model_A.parameters()) + list(model_B.parameters()))
print(params)