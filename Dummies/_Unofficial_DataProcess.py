<<<<<<< HEAD
from torchvision import models

model_A = models.resnet18()
model_B = models.vgg19()

params = set(list(model_A.parameters()) + list(model_B.parameters()))
print(params)
=======
from matplotlib import pyplot as plt


def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    step_i = round(pct * total_steps)
    if step_i <= warmup_steps:  # warm up
        return lr_max * min(1.0, step_i/warmup_steps)
    else:  # decay
        return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr


lr_max = 1e-4
total_steps = 1000000
warmup_steps = 10000
lrs = []

for i in range(total_steps):
    lrs += [linear_warmup_and_then_decay(pct=i/total_steps, lr_max=lr_max, total_steps=total_steps, warmup_steps=warmup_steps)]

plt.plot(range(total_steps), lrs)
plt.show()
>>>>>>> parent of 94e89a3... FINETUNING UPDATE
