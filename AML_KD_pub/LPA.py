import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import cg
import faiss
import scipy.sparse
import scipy.sparse.linalg
import time

# return a knn graph
def navie_knn(dataSet, k):
    """
    :param dataSet: batch_feature N x feat_dim
    :param k:  num neighbor
    :return: knn graph
    """
    numSamples = dataSet.shape[0]  # batch size
    dataSet = F.normalize(dataSet)  # normalize the batch features
    sim_matrix = dataSet.numpy().dot(dataSet.numpy().t())
    # sim_matrix = dataSet.mm(dataSet.t())  # similarity matrix between features

    affinity_matrix = torch.zeros([numSamples, numSamples], dtype=torch.float).cuda()
    for i in range(numSamples):
        _, idx_sorted = torch.sort(sim_matrix[i], descending=True)
        if k > len(idx_sorted):
            k = len(idx_sorted)
        k_neighbor = idx_sorted[0:k]
        affinity_matrix[i][k_neighbor] = 1.

    tmp = affinity_matrix * sim_matrix
    knn_graph = (tmp - torch.diag_embed(torch.diag(tmp)).cuda()) + \
                (tmp - torch.diag_embed(torch.diag(tmp)).cuda()).t()
    return knn_graph


def labelKD(featSetT, featSetS, k, alpha, label, num_class):
    """
    :param featSetT: teacher features
    :param featSetS: student features
    :param k: k neighbor
    :param alpha: balancing weight
    :param label one-hot label
    :return: pseudo labels
    """
    # graph_t = navie_knn(featSetT, k)
    graph_s = navie_knn(featSetS, k)
    # graph_s = F.normalize(navie_knn(featSetS, k))
    numSamples = featSetT.shape[0]
    one_hot_label = torch.zeros(numSamples, num_class).scatter_(1, label.cpu().view(-1, 1), 1).cuda()
    Z = torch.zeros([numSamples, num_class], dtype=torch.float)
    D = torch.sum(graph_s, dim=1)
    D[D == 0] = 1
    D = 1. / torch.sqrt(D)
    D = torch.diag(D)
    Wn = D.mm(graph_s).mm(D)
    A = torch.eye(numSamples).cuda() - alpha * Wn.cuda()
    # A = torch.eye(numSamples).cuda() - alpha * Wn.cuda()
    for i in range(num_class):
        y = one_hot_label[:, i]
        f, _ = cg(A.cpu().detach().numpy(), y.cpu().detach().numpy(), tol=1e-6, maxiter=20)
        Z[:, i] = torch.tensor(f)
    Z[Z < 0] = 0
    Z = F.softmax(Z, dim=1)
    p_label = torch.argmax(Z, dim=1)
    p_conf, _ = torch.max(Z, dim=1)
    correct = (p_label.cpu() == label.cpu()).float()
    acc = correct.mean()
    print('correctly pseudo label is %.2f' % acc)
    return p_label


# extract feature from dataset
def extract_features(train_loader, model):
    model = model.eval()
    embedding_all, label_all, index_all, gt_all, pred_all = [], [], [], [], []
    for i, (x, _, target, index, gt) in enumerate(train_loader):
        x = x.cuda()
        p, _, feat = model(x)
        pred_label = torch.argmax(p, dim=1)
        embedding_all.append(feat.data.cpu())
        label_all.append(target.data.cpu())
        index_all.append(index.data.cpu())
        gt_all.append(gt)
        pred_all.append(pred_label)
    gt_all, pred_all = torch.cat(gt_all), torch.cat(pred_all)
    embedding_all, label_all, index_all = torch.cat(embedding_all), torch.cat(label_all), torch.cat(index_all)
    return embedding_all, label_all, index_all, gt_all, pred_all


#  divide dataset into label and unlabel samples
def divide_label_unlable(train_loader, model_s, model_t):
    model_s.eval()
    model_t.eval()
    labeled_all, unlabeled_all, label, indices = [], [], [], []
    for i, (x, _, target, index, gt) in enumerate(train_loader):
        x = x.cuda()
        with torch.no_grad():
            ps, _, _ = model_s(x)
            pt, _, _ = model_t(x)
            pred_s = torch.argmax(ps, dim=1).cpu()
            pred_t = torch.argmax(pt, dim=1).cpu()
            tmp = (pred_t == target)
            # label_idx = ((pred_s == target) * (pred_t == target)).nonzero().squeeze()
            label_idx = (pred_t == target).nonzero()
            unlable_idx = (~(pred_t == target)).nonzero()
            # unlable_idx = (~((pred_s == target) * (pred_t == target))).nonzero().squeeze()
            labeled_all.append(index[label_idx])
            unlabeled_all.append(index[unlable_idx])
            label.append(target[label_idx])
            indices.append(index)

    indices = torch.cat(indices)
    labeled_all, unlabeled_all, label = torch.cat(labeled_all), torch.cat(unlabeled_all), torch.cat(label)
    return labeled_all, unlabeled_all, label, indices


def LPA(train_loader, model_s, model_t, args):
    start = time.time()
    feats, labels, indices, gts, preds = extract_features(train_loader, model_s)  # obtain features and labels
    labeled_all, unlable_all, label, indices2 = divide_label_unlable(train_loader, model_s, model_t)

    print('labeled number is %d, unlabeled number is %d' % (len(labeled_all), len(unlable_all)))
    feats = feats.numpy()
    N = feats.shape[0]
    new_idx = np.argsort(indices)
    new_label = labels[new_idx]  # the new label
    gts = gts[new_idx]
    feats = feats[new_idx]
    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, gts.cpu().view(-1, 1), 1).numpy()
    d = feats.shape[1]  # feature dim
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(feats)
    D, I = index.search(feats, args.k + 1)
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (args.k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply
    # label propagation
    Z = np.zeros((N, args.num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - args.alpha_ * Wn
    for i in range(args.num_class):
        y = one_hot_label[:, i]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.softmax(torch.tensor(Z), 1).numpy()
    p_labels = np.argmax(probs_l1, 1)
    correct = (p_labels == gts.cpu().numpy())
    end_time = time.time()
    print('label precision for lpa is %.3f, time costs %.2fs for computing pseudo labels' % (
        correct.mean(),  end_time - start))
    return probs_l1, p_labels
