
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist
import numpy as np
import time
import os
import dgl
from sklearn.metrics import roc_curve, auc, average_precision_score
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def cos_sim(args, ori_adj, feat_np):
    sim = cosine_similarity(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)
    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) > np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]
    edge_assign[pos_ind] = 1.0
    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    AUC, AP = recon_metric(ori_adj, sim_)
    print("SQ3L -- cos_sim. AUC: %f AP: %f" % (AUC, AP))


def cos_dist(args, ori_adj, feat_np):
    sim = cdist(feat_np, feat_np, 'cosine')
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)
    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]
    edge_assign[pos_ind] = 1.0
    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    AUC, AP = recon_metric(ori_adj, sim_)
    print("SQ3L -- cos_sim. AUC: %f AP: %f" % (AUC, AP))

def euclidean_dis(args, ori_adj, feat_np):
    sim = euclidean_distances(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)
    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0
    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    # print((time_end-time_start)/60)

    AUC, AP = recon_metric(ori_adj, sim_)
    print("SQ3L -- cos_sim. AUC: %f AP: %f" % (AUC, AP))


def man_dis(args, ori_adj, feat_np):
    sim = manhattan_distances(feat_np)
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)
    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0
    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()
    AUC, AP = recon_metric(ori_adj, sim_)
    print("SQ3L -- cos_sim. AUC: %f AP: %f" % (AUC, AP))

def corr_dis(args, ori_adj, feat_np):
    sim = cdist(feat_np, feat_np, 'correlation')
    node_dist = np.expand_dims(sim[np.triu_indices(sim.shape[0])], axis=1)
    edge_assign = np.squeeze(np.zeros_like(node_dist), axis=1)


    time_start = time.time()
    initial_centers = kmeans_plusplus_initializer(node_dist, 2).initialize()
    kmeans_instance = kmeans(node_dist, initial_centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    clusters1 = node_dist[clusters[0]]
    clusters2 = node_dist[clusters[1]]
    if np.mean(clusters1) < np.mean(clusters2):
        pos_ind = clusters[0]
    else:
        pos_ind = clusters[1]

    edge_assign[pos_ind] = 1.0
    sim_ = np.zeros((sim.shape[0], sim.shape[0]))
    tri_ind = np.triu_indices(len(sim_))
    sim_[tri_ind] = edge_assign
    sim_ = sim_ + sim_.T
    np.fill_diagonal(sim_, 1)
    time_end = time.time()

    AUC, AP = recon_metric(ori_adj, sim_)
    print("SQ3L -- cos_sim. AUC: %f AP: %f" % (AUC, AP))


def recon_metric(ori_adj, inference_adj):
    real_edge = ori_adj.reshape(-1)
    pred_edge = inference_adj.reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)

    AUC = auc(fpr, tpr)
    AP = average_precision_score(real_edge, pred_edge)
    return AUC, AP




