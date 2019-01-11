import torch
a = torch.tensor([[1,2,3,4],[2,5,6,7.0],[3,5,6,7.0],[4,5,6,7.0]])
b = torch.tensor([[1,2,3,4],[4,5,6,7.0]])

c = torch.tensor([1,5,3,4])
k = c>3
print(k)
k1 = c<5
print()
print(c[k*k1])
print(a[k*k1])
kk = torch.randperm(c.size(0))
print(kk)
idx = kk[:2]
sam = c[idx]
print(sam)