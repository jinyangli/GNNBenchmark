import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import RedditDataset 

class SPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, x):
        adj, adj_T = graph
        ctx.save_for_backward(adj_T)
        return torch.spmm(adj, x)

    @staticmethod
    def backward(ctx, grad):
        adj_T, = ctx.saved_tensors
        return None, torch.spmm(adj_T, grad)

spmm = SPMM.apply

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._linear = nn.Linear(in_feats, out_feats)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self._linear.weight, gain=gain)
        nn.init.zeros_(self._linear.bias)

    def forward(self, graph, feat):
        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            rst = spmm(graph, self._linear(feat))
        else:
            # aggregate first then mult W
            rst = self._linear(spmm(graph, feat))
        if self._activation is not None:
            rst = self._activation(rst)

        return rst

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, adjs, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(adjs, h)
        return h

def evaluate(model, adjs, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adjs, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def build_sparse_matrix(g, device):
    # normalization
    n = g.number_of_nodes()
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm
    g.apply_edges(fn.u_mul_v("norm", "norm", "weight"))
    weight = g.edata.pop("weight")
    src, dst = g.all_edges()
    adj = torch.sparse.FloatTensor(torch.stack([dst, src]), weight, torch.Size([n,n]))
    adj = adj.coalesce().to(device)
    adj_T = torch.sparse.FloatTensor(torch.stack([src, dst]), weight, torch.Size([n,n]))
    adj_T = adj_T.coalesce().to(device)
    return adj, adj_T

def main(args):
    # load and preprocess dataset
    data = RedditDataset(self_loop=True)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    # graph preprocess and calculate normalization factor
    adjs = build_sparse_matrix(data.graph, device)

    # create GCN model
    model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    model.to(device)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(adjs, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, adjs, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}" \
                .format(epoch, np.mean(dur), loss.item(), acc))

    print()
    acc = evaluate(model, adjs, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN on Reddit dataset')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
