import torch

mean = torch.rand(1, 2)
std = torch.rand(1, 2)

torch.manual_seed(1)
print(torch.normal(mean, std))
