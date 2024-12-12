import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv, GINConv, SGConv, SAGEConv
from torch.nn.functional import relu, elu
from dgl.nn.pytorch.glob import AvgPooling
from dgl.ops import segment
import dgl


class SSL(nn.Module):

    def __init__(self, in_dim, out_dim, encoder='gcn'):
        super(SSL, self).__init__()

        self.encoder = encoder

        if self.encoder == 'gcn':
            self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
            self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())

        elif self.encoder == 'gat':
            self.encoder1 = GATConv(in_dim, out_dim, num_heads=1, bias=False, activation=None)
            self.encoder2 = GATConv(in_dim, out_dim, num_heads=1, bias=False, activation=None)

        elif self.encoder == 'gin':
            self.lin = th.nn.Linear(in_dim, out_dim)
            self.encoder1 = GINConv(self.lin, aggregator_type='max', activation=relu)
            self.encoder2 = GINConv(self.lin, aggregator_type='max', activation=relu)

        elif self.encoder == 'sgc':
            self.encoder1 = SGConv(in_dim, out_dim, k=1, norm=relu, bias=True)
            self.encoder2 = SGConv(in_dim, out_dim, k=1, norm=relu, bias=True)

        elif self.encoder == 'sage':
            self.encoder1 = SAGEConv(in_dim, out_dim, norm=relu, bias=True, aggregator_type='pool')
            self.encoder2 = SAGEConv(in_dim, out_dim, norm=relu, bias=True, aggregator_type='pool')
        self.pooling = AvgPooling()
        self.disc = BIlinearscore(out_dim)
        self.activate = nn.Sigmoid()

    def get_embedding(self, diff_graph_1, diff_graph_2, feat, edge_weight_1, edge_weight_2):
        if self.encoder == 'gcn':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)
        elif self.encoder == 'gat':
            h1 = self.encoder1(diff_graph_1, feat)
            h2 = self.encoder2(diff_graph_2, feat)
        elif self.encoder == 'gin':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)
        elif self.encoder == 'sgc':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)
        elif self.encoder == 'sage':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)
        return (h1 + h2).detach()


    def forward(self, diff_graph_1, diff_graph_2, feat, shuf_feat, edge_weight_1, edge_weight_2):
        # GCN
        if self.encoder == 'gcn':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)

            h3 = self.encoder1(diff_graph_1, shuf_feat, edge_weight=edge_weight_1)  # negative representation
            h4 = self.encoder2(diff_graph_2, shuf_feat, edge_weight=edge_weight_2)

        elif self.encoder == 'gat':
            h1 = self.encoder1(diff_graph_1, feat)
            h2 = self.encoder2(diff_graph_2, feat)

            h3 = self.encoder1(diff_graph_1, shuf_feat)
            h4 = self.encoder2(diff_graph_2, shuf_feat)

        elif self.encoder == 'gin':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)

            h3 = self.encoder1(diff_graph_1, shuf_feat, edge_weight=edge_weight_1)
            h4 = self.encoder2(diff_graph_2, shuf_feat, edge_weight=edge_weight_2)

        elif self.encoder == 'sgc':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)

            h3 = self.encoder1(diff_graph_1, shuf_feat, edge_weight=edge_weight_1)  # negative representation
            h4 = self.encoder2(diff_graph_2, shuf_feat, edge_weight=edge_weight_2)

        elif self.encoder == 'sage':
            h1 = self.encoder1(diff_graph_1, feat, edge_weight=edge_weight_1)
            h2 = self.encoder2(diff_graph_2, feat, edge_weight=edge_weight_2)

            h3 = self.encoder1(diff_graph_1, shuf_feat, edge_weight=edge_weight_1)  # negative representation
            h4 = self.encoder2(diff_graph_2, shuf_feat, edge_weight=edge_weight_2)


        c1 = self.activate(self.pooling(diff_graph_1, h1))
        c2 = self.activate(self.pooling(diff_graph_2, h2))
        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out


class BIlinearscore(nn.Module):
    def __init__(self, dim):
        super(BIlinearscore, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        g_x1 = c1.expand_as(h1).contiguous()  # graph-repr 1
        g_x2 = c2.expand_as(h2).contiguous()  # graph-repr 2
        # positive
        aln_1 = self.fn(h2, g_x1).squeeze(1)
        aln_2 = self.fn(h1, g_x2).squeeze(1)
        # negative
        aln_3 = self.fn(h4, g_x1).squeeze(1)
        aln_4 = self.fn(h3, g_x2).squeeze(1)
        logits = th.cat((aln_1, aln_2, aln_3, aln_4))
        return logits



class GCN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, h_dim)
        self.conv2 = GraphConv(h_dim, h_dim)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h1 = F.relu(h)
        h2 = self.conv2(g, h1)
        return h1, h2




# --------------------------GAE--------------------------
class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super(GAE, self).__init__()
        self.GNN = GCN(in_dim, hidden_dims, hidden_dims)
        self.decoder = InnerProductDecoder(activation=lambda x: x)

    def forward(self, g, in_feat):
        g.ndata['h'] = in_feat
        h = g.ndata['h']
        h = self.GNN(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['h']
        h = self.GNN(g, h)
        return h

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj
# --------------------------GAE--------------------------








