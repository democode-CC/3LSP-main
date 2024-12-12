import argparse
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from dataset import data_augment
from model import SSL
from util import set_seed
from psc_metric import cos_sim, cos_dist, euclidean_dis, man_dis, corr_dis



parser = argparse.ArgumentParser(description='SSL')

# Random seed
parser.add_argument('--seed', type=int, default=100, help='random seed.')
parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset. cora, citeseer, pubmed, reddit')
parser.add_argument('--partition_method', type=str, default='metis', help='adopted partition method.')
parser.add_argument('--subgraph_ratio', type=float, default=0.5, help='size ratio of sampled subgraph.')
parser.add_argument('--adj_init', type=str, default='cos_sim', help='adj initialize method. random, zero, cos_sim, attention, ori, full')
parser.add_argument('--adj_init_random_p', type=float, default='0.5', help='Probability for edge creation when using random')

parser.add_argument('--alpha1', type=float, default=0.2, help='alpha1 for ppr.')
parser.add_argument('--alpha2', type=float, default=0.4, help='alpha2 for ppr.')
parser.add_argument('--sim_k', type=int, default=5, help='5, 10, 20, 50, 100, 200.')
parser.add_argument('--encoder', type=str, default='gcn', help='encoder GNN.')



parser.add_argument('--gpu', type=int, default=1, help='GPU index. Default: -1, using cpu.')
parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of SSL.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of SSL.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')



args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

# set seed
# set_seed(args.seed)


if __name__ == '__main__':
    print(args)

    # DATA PREPARATION
    target_graph, diff_graph_1, diff_graph_2, target_feat, target_labels, target_train_mask, target_val_mask, target_test_mask, edge_weight_1, edge_weight_2, target_n_classes = data_augment(args.dataset, args.epsilon, args.partition_method, args.subgraph_ratio, args.adj_init, args.adj_init_random_p, args.alpha1, args.alpha2, args.sim_k)
    target_graph = target_graph.to(args.device)
    target_feat = target_feat.to(args.device)
    target_labels = target_labels.to(args.device)
    target_train_mask = target_train_mask.to(args.device)
    target_val_mask = target_val_mask.to(args.device)
    target_test_mask = target_test_mask.to(args.device)
    target_n_node = target_graph.number_of_nodes()
    target_adj = target_graph.adj().to_dense().to(args.device)
    target_adj[target_adj == 2] = 1
    target_adj_np = target_adj.cpu().numpy()
    n_feat = target_feat.shape[1]
    diff_graph_1 = diff_graph_1.to(args.device)
    edge_weight_1 = torch.tensor(edge_weight_1).float().to(args.device)
    diff_graph_2 = diff_graph_2.to(args.device)
    edge_weight_2 = torch.tensor(edge_weight_2).float().to(args.device)
    feat = target_feat.to(args.device)
    n_node = target_graph.number_of_nodes()
    lbl1 = torch.ones(n_node * 2)
    lbl2 = torch.zeros(n_node * 2)
    lbl = torch.cat((lbl1, lbl2))  # 5544
    lbl = lbl.to(args.device)

    # MODEL INITIALIZATION
    model = SSL(n_feat, args.hid_dim, encoder=args.encoder)
    model = model.to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_logits = nn.MSELoss()
    best = float('inf')
    cnt_wait = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_list = []

    # TRAINING
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Contrastive Learning
        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)
        out = model(diff_graph_1, diff_graph_2, feat, shuf_feat, edge_weight_1, edge_weight_2)
        if args.encoder == 'gat':
            out = out.squeeze()
        loss = loss_fn(out, lbl)  # Contrastive Loss
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            torch.save(model.state_dict(), 'checkpoint/best_model.pkl')




    # NAO-LP EVALUATION
    print('-------------------------------------Our results (SQ3L)----------------------------------------')
    best_model = torch.load('checkpoint/best_model.pkl')
    embeds = model.get_embedding(diff_graph_1, diff_graph_2, feat, edge_weight_1, edge_weight_2)
    embeds = embeds.cpu().numpy()

    cos_sim(args, target_adj_np, embeds)
    cos_dist(args, target_adj_np, embeds)
    euclidean_dis(args, target_adj_np, embeds)
    man_dis(args, target_adj_np, embeds)
    corr_dis(args, target_adj_np, embeds)
    print('\n'*3)


    #--------------------Baseline-----------------------------------------------------------------------------------
    print('-------------------------------------Baselines (PNASC)----------------------------------------')
    target_feat_np = target_feat.detach().cpu().numpy()
    cos_sim(args, target_adj_np, target_feat_np)
    cos_dist(args, target_adj_np, target_feat_np)
    euclidean_dis(args, target_adj_np, target_feat_np)
    man_dis(args, target_adj_np, target_feat_np)
    corr_dis(args, target_adj_np, target_feat_np)
    print('\n'*3)



