import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import load_data 

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
                 activation=None,
                 norm=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self._activation = activation
        self.norm = norm
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, graph, feat):
        if self.norm is not None:
            feat = feat * self.norm[0]
        feat = torch.matmul(feat, self.weight)
        rst = spmm(graph, feat)
        if self.norm is not None:
            feat = feat * self.norm[1]
        rst = rst + self.bias
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
                 dropout,
                 norm):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=norm))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=norm))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm=norm))
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
    n = g.number_of_nodes()
    src, dst = g.all_edges()
    weight = torch.ones(src.shape)
    adj = torch.sparse.FloatTensor(torch.stack([dst, src]), weight, torch.Size([n, n]))
    adj = adj.coalesce().to(device)
    adj_T = torch.sparse.FloatTensor(torch.stack([src, dst]), weight, torch.Size([n, n]))
    adj_T = adj_T.coalesce().to(device)
    return adj, adj_T

def main(args):
    data = load_data(args)
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

    g = data.graph
    norm_out = g.out_degrees().float().clamp(min=1).pow(-0.5).view(-1, 1)
    norm_in = g.in_degrees().float().clamp(min=1).pow(-0.5).view(-1, 1)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        norm_out = norm_out.to(device)
        norm_in = norm_in.to(device)

    norm = norm_out, norm_in
    adjs = build_sparse_matrix(g, device)

    # create GCN model
    model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                norm=norm)
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
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="reddit-self-loop")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()
    print(args)

    main(args)
