import torch

torch.set_default_device('cuda:0')

a = torch.ones((2, 3), dtype=torch.int32)

print(a)

out = torch.reshape(a, (3, 2))
print(out)

print(torch.t(out))

print(torch.transpose(out, 0, 1))

print(torch.unsqueeze(a, 0))
print(torch.unsqueeze(a, -1))
print(torch.squeeze(a))
print(torch.unbind(a, dim=0))
print(torch.flip(a, dims=[0, 1]))

print(torch.rot90(a))
print(torch.rot90(a, 2))
