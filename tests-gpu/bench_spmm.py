import argparse
import time
import numpy as np
import torch
from dgl.data import load_data 



def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"

    g = load_data(args).graph
    n_nodes = g.number_of_nodes()

    # build sparse matrix
    src, dst = g.all_edges()
    adj = torch.sparse.FloatTensor(torch.stack([dst, src]), 
                                   torch.ones(src.shape), 
                                   torch.Size([n_nodes, n_nodes]))
    adj = adj.coalesce().to(device)

    # generate features
    features = torch.randn(n_nodes, args.n_hidden).to(device)

    # warm up
    for _ in range(args.n_repeat):
        x = torch.spmm(adj, features)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.n_repeat):
        x = torch.spmm(adj, features)
    torch.cuda.synchronize()
    end = time.time()
    print("Time (ms): {:.3f}".format((end - start) * 1e3 / args.n_repeat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SPMM')
    parser.add_argument("--dataset", type=str, default="reddit-self-loop")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-repeat", type=int, default=100)
    args = parser.parse_args()
    print(args)

    main(args)
