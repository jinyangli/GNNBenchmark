import networkx as nx
import argparse
import dgl
from dgl.data import register_data_args, load_data

import graph_io

def main(args):
    data = load_data(args)
    g = data.graph
    if isinstance(g, dgl.DGLGraph):
        csr = g.adjacency_matrix_scipy(transpose=True)
    else:
        csr = nx.to_scipy_sparse_matrix(g, weight=None, format='csr')

    graph_io.save_graph(args.out, csr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, help='Output file name')
    register_data_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
