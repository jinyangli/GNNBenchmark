import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--n-repeat", type=int, default=100)
parser.add_argument("--matrix-size", type=int, default=1000)
args = parser.parse_args()

if args.gpu < 0:
    device = "cpu"
else:
    device = args.gpu

size = args.matrix_size
a = torch.randn(size, size).to(device)
b = torch.randn(size, size).to(device)
old_a = a

repeat = args.n_repeat

# warm up
for _ in range(repeat):
    a = torch.matmul(a, b)

a = old_a

if device != "cpu":
    torch.cuda.synchronize()
t0 = time.time()

for _ in range(repeat):
    a = torch.matmul(a, b)

if device != "cpu":
    torch.cuda.synchronize()
t1 = time.time()

print("{:.3f}".format((t1 - t0) / repeat * 1000))
