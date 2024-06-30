# 重点掌握，abs(),sign(),sigmoid()

import torch

a = torch.rand(3,4)
print(a)
print(torch.abs(a))
print(torch.sign(a))
print(torch.sigmoid(a))