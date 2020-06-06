import torch
import time

a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)
old_a = a

repeat = 100

# warm up
for _ in range(repeat):
    a = torch.matmul(a, b)

a = old_a
t0 = time.time()

for _ in range(repeat):
    a = torch.matmul(a, b)

t1 = time.time()
print((t1 - t0) / repeat)
