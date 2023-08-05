import argparse
import time
import torch.nn.functional as F
from torch import optim
from utils import *
import os
import faiss
import torch
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from models.weight_net import WeightNet
from LPA import extract_features
from utils_lpa import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--kd_T', default=4, type=float, help='kd temperature')
parser.add_argument('--epoch', default=240, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=5e-4)
parser.add_argument('--warm_epoch', default=20)
parser.add_argument('--seed', default=0)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='/data/jrq/cifar-100-python/', type=str,
                    help='path to dataset')
parser.add_argument('--arch_t', default='wrn_40_2', type=str)
parser.add_argument('--arch_s', default='wrn_16_2', type=str)
parser.add_argument('--data_root', default='/data/jrq/cifar100/', type=str)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--alpha_', default=0.99, type=int)
parser.add_argument('--k', default=20, type=int, help='k-neighbors')
parser.add_argument('--gpu', default="3", type=str, help='gpuid')
parser.add_argument('--t-path', type=str, default='')  # teacher checkpoint path

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
peusdo_acc = []

def sche_alpha(e):
    return 0.99

def lpa(train_loader, model):
    start = time.time()
    feats, labels, indices, gts, preds = extract_features(train_loader, model)  # obtain features and labels
    feats = feats.numpy()
    N = feats.shape[0]
    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, preds.cpu().view(-1, 1), 1).numpy()
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

   
    # label propagation
    Z = np.zeros((N, args.num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - args.alpha_ * Wn
    for i in range(args.num_class):
        y = one_hot_label[:, i]
        y /= float(y.sum())
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
        Z[:, i] = f

    # Handle numberical errors
    # Z[Z < 0] = 0
    labels = labels.numpy()
    
    probs_l1 = F.softmax(torch.tensor(Z), 1).numpy()

    conf = np.max(probs_l1, axis=1)
    # meadian = np.median(conf)
    trsd = np.percentile(conf, 5)
    conf[conf < trsd] = 0
    relabel_idx = conf.nonzero()
    p_labels = np.argmax(probs_l1, 1)
    labels[relabel_idx] = p_labels[relabel_idx]

    new_idx = np.argsort(indices)
    new_label = labels[new_idx]  # the new label
   
    correct = (labels == gts.cpu().numpy())
    correct_pred = (preds.cpu().numpy() == gts.cpu().numpy())
    # new_label = preds.cpu().numpy()[new_idx]
    end_time = time.time()
    print('label precision for lpa is %.3f, for pred is %.3f time costs %.2fs for computing pseudo labels' % (
        correct.mean(), correct_pred.mean(), end_time - start))
    return new_label

def lpa_by_student_feat(info_dict, new_label, train_loader, model_s):  # student feat + teacher label
    label_pred = info_dict['label']
    index_keep = info_dict['easy_index']
    feats, labels, indices, gts, preds = extract_features(train_loader, model_s)
    feats = F.normalize(feats, dim=1)
    N = feats.shape[0]
    corr_stu = (label_pred[indices].cpu().numpy() == gts.cpu().numpy()).mean()
    print('label precision from student is %.3f' % corr_stu)
    feats = feats.numpy()

    d = feats.shape[1]  # feature dim

    class_center = setup_class_center(model_s, train_loader, label_pred.cpu().numpy())
    weight_all, index_all = [], []

    model_s.eval()
    for _, (x_light, _, target, index, gt) in enumerate(train_loader):
        x_light = x_light.cuda()
        new_target = label_pred[index.numpy()].cuda()  # replace the original targets
        # foward
        _, _, fs = model_s(x_light)
        score = torch.mm(F.normalize(fs.detach().cuda(), dim=1), F.normalize(class_center, dim=1).t().cuda())
        w = torch.sigmoid(score.gather(1, new_target.view(-1, 1))).squeeze().detach()
        weight_all.append(w)
        index_all.append(index)
    weight_all, index_all = torch.cat(weight_all), torch.cat(index_all)
    idx_sort = np.argsort(index_all.cpu().numpy())
    weight = weight_all[idx_sort].unsqueeze(1)
    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, torch.from_numpy(new_label)[indices].view(-1, 1),
                                                            1).numpy()
    one_hot_weight = torch.zeros(N, args.num_class).scatter_(1, torch.from_numpy(new_label)[indices].cpu().view(-1, 1),
                                                             weight.cpu())

    one_hot_label = one_hot_weight * one_hot_label
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

    # label propagation
    Z = np.zeros((N, args.num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - args.alpha_ * Wn
    for i in range(args.num_class):
        y = one_hot_label[:, i]
        y /= float(y.sum())
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    probs_l1 = F.softmax(torch.tensor(Z), 1).numpy()
    conf = np.max(probs_l1, axis=1)
    meadian = np.percentile(conf, 0)
    conf[conf < meadian] = 0
    relabel_idx = conf.nonzero()
    p_labels = np.argmax(probs_l1, 1)
    # labels = p_labels
    labels = labels.numpy()
    labels[relabel_idx] = p_labels[relabel_idx]
    new_idx = np.argsort(indices)
    new_label = labels[new_idx]  # the new label
    new_label[index_keep] = label_pred[index_keep].cpu().numpy()
    return new_label


def lpa_by_teacher_feat(info_dict, new_label, train_loader, model_t, model_s):
    label_pred = info_dict['label']
    index_keep = info_dict['easy_index']
    feats, labels, indices, gts, preds = extract_features(train_loader, model_t)
    feats = F.normalize(feats, dim=1)
    N = feats.shape[0]
    feats = feats.numpy()
    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, torch.from_numpy(new_label)[indices].view(-1, 1),
                                                            1).numpy()
    d = feats.shape[1]  # feature dim

    class_center = setup_class_center(model_s, train_loader, label_pred.cpu().numpy())
    weight_all, index_all = [], []

    model_s.eval()
    for _, (x_light, _, _, index, _) in enumerate(train_loader):
        x_light = x_light.cuda()
        new_target = label_pred[index.numpy()].cuda()  # replace the original targets
        # foward
        _, _, fs = model_s(x_light)
        score = torch.mm(F.normalize(fs.detach().cuda(), dim=1), F.normalize(class_center, dim=1).t().cuda())
        w = torch.sigmoid(score.gather(1, new_target.view(-1, 1))).squeeze().detach()
        weight_all.append(w)
        index_all.append(index)
    weight_all, index_all = torch.cat(weight_all), torch.cat(index_all)
    idx_sort = np.argsort(index_all.cpu().numpy())
    weight = weight_all[idx_sort].unsqueeze(1)

    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, torch.from_numpy(new_label)[indices].view(-1, 1),
                                                            1).numpy()
    one_hot_weight = torch.zeros(N, args.num_class).scatter_(1, torch.from_numpy(new_label)[indices].cpu().view(-1, 1),
                                                             weight.cpu())

    one_hot_label = one_hot_weight * one_hot_label

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

    # label propagation
    Z = np.zeros((N, args.num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - args.alpha_ * Wn
    for i in range(args.num_class):
        y = one_hot_label[:, i]
        y /= float(y.sum())
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    probs_l1 = F.softmax(torch.tensor(Z), 1).numpy()
    conf = np.max(probs_l1, axis=1)
    meadian = np.percentile(conf, 0)
    conf[conf < meadian] = 0
    relabel_idx = conf.nonzero()
    p_labels = np.argmax(probs_l1, 1)
    labels = labels.numpy()
    labels[relabel_idx] = p_labels[relabel_idx]
    new_idx = np.argsort(indices)
    new_label = labels[new_idx]  # the new label
    new_label[index_keep] = label_pred[index_keep].cpu().numpy()
    return new_label


def iterate_lpa(info_dict, train_loader, model, model_s, weight_net, e):
    start = time.time()
    label_pred = info_dict['label']
    index_keep = info_dict['easy_index']
    feats, labels, indices, gts, preds = extract_features(train_loader, model)  # obtain features and labels
    N = feats.shape[0]
    class_center = setup_class_center(model_s, train_loader, label_pred.cpu().numpy())
    weight_all, index_all = [], []
    weight = torch.ones(N, args.num_class)
    if e >= 10:
        model_s.eval()
        weight_net.eval()
        for _, (x_light, _, _, index, _) in enumerate(train_loader):
            x_light = x_light.cuda()
            batch_size = x_light.shape[0]
            new_target = label_pred[index.numpy()].cuda()  # replace the original targets
            one_hot_new = torch.zeros(batch_size, 100).scatter_(1, new_target.cpu().view(-1, 1), 1).cuda()
            _, _, fs = model_s(x_light)
            score = torch.mm(F.normalize(fs.detach().cuda(), dim=1), F.normalize(class_center, dim=1).t().cuda())
            w_in = torch.cat((score, one_hot_new), dim=1).detach().cuda()
            w = weight_net(w_in).detach()
            weight_all.append(w)
            index_all.append(index)
        weight_all, index_all = torch.cat(weight_all), torch.cat(index_all)
        idx_sort = np.argsort(index_all.cpu().numpy())
        weight = weight_all[idx_sort]
    feats = feats.numpy()
    one_hot_label = torch.zeros(N, args.num_class).scatter_(1, label_pred[indices].cpu().view(-1, 1), 1).numpy()
    one_hot_weight = torch.zeros(N, args.num_class).scatter_(1, label_pred[indices].cpu().view(-1, 1), weight.cpu())
    one_hot_label = one_hot_weight * one_hot_label
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
    # label propagation
    Z = np.zeros((N, args.num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - args.alpha_ * Wn
    for i in range(args.num_class):
        y = one_hot_label[:, i]
        y /= float(y.sum())
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
        Z[:, i] = f
    # Handle numberical errors
    Z[Z < 0] = 0
    probs_l1 = F.softmax(torch.tensor(Z), 1).numpy()
    conf = np.max(probs_l1, axis=1)
    meadian = np.percentile(conf, 0)
    conf[conf < meadian] = 0
    relabel_idx = conf.nonzero()
    p_labels = np.argmax(probs_l1, 1)
    # labels = p_labels
    labels = labels.numpy()
    labels[relabel_idx] = p_labels[relabel_idx]
    new_idx = np.argsort(indices)
    new_label = labels[new_idx]  # the new label
    new_label[index_keep] = label_pred[index_keep].cpu().numpy()
    gts_tmp = gts.cpu().numpy()[new_idx]
    correct = (new_label == gts_tmp)
    end_time = time.time()
    print('label precision: %.2f || time %.2fs' % (
        correct.mean(),  end_time - start))
    return new_label


def warm_up(train_loader, model_s, model_t, optimizer, new_target):
    model_s.train()
    model_t.eval()
    for _, (x_light, _, target, index, _) in enumerate(train_loader):
        x_light = x_light.cuda()
        target = torch.from_numpy(new_target[index]).cuda()
        batch_size = target.size(0)

        #  mix-up
        ps, _, _ = model_s(x_light)
        rand_idx = torch.randperm(batch_size)
        l = np.random.beta(args.alpha, args.alpha)
        x_mix = l * x_light + (1 - l) * x_light[rand_idx]

        # forward
        ps_mix, _, _ = model_s(x_mix)
        ps_mix = F.log_softmax(ps_mix / args.kd_T, dim=1)
        with torch.no_grad():
            pt_mix, _, _ = model_t(x_mix)
            pt_mix = F.softmax(pt_mix / args.kd_T, dim=1)

        # compute loss
        one_hot_target = torch.zeros(batch_size, 100).scatter_(1, target.cpu().view(-1, 1), 1).cuda()
        one_hot_shuffle = torch.zeros(batch_size, 100).scatter_(1, target[rand_idx].cpu().view(-1, 1), 1).cuda()
        l_ce = F.cross_entropy(ps, target)
        l_ces = cross_entropy(ps_mix, one_hot_target) * l + cross_entropy(ps_mix, one_hot_shuffle) * (1 - l)
        l_kd = 0.9 * args.kd_T * args.kd_T * F.kl_div(ps_mix, pt_mix, reduction='batchmean')
        loss = l_ce + l_kd + l_ces
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def setup_refined_label(model_s, train_loader, info_dict):
    for _, (x_light, _, target, index, _) in enumerate(train_loader):
        x_light, target = x_light.cuda(), target.cuda()
        ratio = 1 - args.r if args.noise_mode is not None else 1.
        batch_size = target.size(0)
        with torch.no_grad(): 
            ps, _, _ = model_s(x_light)
            pred_all = F.softmax(ps, dim=1)
            conf, _ = torch.max(pred_all, dim=1)
            pred_all = torch.argmax(pred_all, dim=1)
            loss_pack = F.cross_entropy(ps, target, reduction='none')
            ind_sorted = np.argsort(loss_pack.cpu().numpy())
            ind_keep = ind_sorted[:int(ratio * batch_size)]
            ind_relabel = ind_sorted[int(ratio * batch_size):]
            info_dict['label'].append(target[ind_keep])
            info_dict['easy_index'].append(index[ind_keep])
            info_dict['index'].append(index[ind_keep])
            info_dict['label'].append(pred_all[ind_relabel])
            info_dict['index'].append(index[ind_relabel])
            info_dict['conf'].append(conf)
    info_dict['label'] = torch.cat(info_dict['label'])
    info_dict['index'] = torch.cat(info_dict['index'])
    info_dict['conf'] = torch.cat(info_dict['conf'])
    info_dict['easy_index'] = torch.cat(info_dict['easy_index'])
    ind_sort = np.argsort(info_dict['index'])
    info_dict['index'] = info_dict['index'][ind_sort]
    info_dict['label'] = info_dict['label'][ind_sort]
    info_dict['conf'] = info_dict['conf'][ind_sort]
    return info_dict


def setup_class_center(model, train_loader, new_label):
    center = [0] * args.num_class
    feat_all, label_all = [], []
    model.eval()
    for _, (x_light, _, _, index, _) in enumerate(train_loader):
        x_light = x_light.cuda()
        with torch.no_grad():
            _, _, fs = model(x_light)
            feat_all.append(fs.detach())
            label_all.append(torch.from_numpy(new_label[index.numpy()]))

    feat_all, label_all = torch.cat(feat_all), torch.cat(label_all)
    ind_sorted = np.argsort(label_all.cpu().numpy())
    feat_all = feat_all.cpu().numpy()[ind_sorted]
    cursor = 0
    for i in range(args.num_class):
        num_i = np.sum(label_all.cpu().numpy() == i)
        center_i_mean = np.sum(feat_all[cursor:cursor + num_i], axis=0) / float(num_i)
        center[i] = torch.from_numpy(center_i_mean).unsqueeze(1)
        cursor += num_i
    return torch.cat(center, dim=1).t()


def main():
    model_s, model_t = create_model(args)
    weight_net = WeightNet(class_num=args.num_class).cuda()
    optimizer_w = optim.SGD(weight_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_w = MultiStepLR(optimizer_w, milestones=args.milestones, gamma=args.gamma)

    # train_loader, test_loader = create_dataloader_student(args)
    train_loader, test_loader = create_a2adataloader(args)
    try:
        state_dict = torch.load(args.t_path)['model_t']
        model_t.backbone.load_state_dict(state_dict)
    except Exception as e:
        print('can not load teacher model')
    print('teacher acc:')
    test(model_t, test_loader=test_loader, epoch=None, args=args)
    new_label = lpa(train_loader, model_t)
    optimizer = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    model_t.eval()
    for epoch in range(args.epoch):
        model_s.train()
        if epoch < args.warm_epoch:
            warm_up(train_loader, model_s, model_t, optimizer, new_label)
        else:
            info_dict = {
                'index': [],
                'label': [],
                'conf': [],
                'easy_index': []
            }
            info_dict = setup_refined_label(model_s, train_loader, info_dict)
            if epoch <= 180:
                new_label = iterate_lpa(info_dict, train_loader, model_t, model_s, weight_net, epoch)
            else:
                if epoch % 2 == 0:
                    new_label = lpa_by_student_feat(info_dict, new_label, train_loader, model_s)
                else:
                    new_label = lpa_by_teacher_feat(info_dict, new_label, train_loader, model_t, model_s)

            class_center = setup_class_center(model_s, train_loader, new_label)  # obtain the class center feature
            model_s.train()
            weight_net.train()

            for _, (x_light, _, target, index, gt) in enumerate(train_loader):
                x_light = x_light.cuda()
                new_target = torch.from_numpy(new_label[index.numpy()]).cuda()  # replace the original targets
                batch_size = target.size(0)

                #  mix-up
                l = np.random.beta(args.alpha, args.alpha)
                rand_idx = torch.randperm(batch_size)
                x_shuffle = x_light[rand_idx]
                x_mix = x_light * l + (1 - l) * x_shuffle

                # foward
                ps, _, fs = model_s(x_light)
                ps_mix, _, _ = model_s(x_mix)
                ps_mix = F.log_softmax(ps_mix / args.kd_T, dim=1)
                score = torch.mm(F.normalize(fs.detach().cuda(), dim=1), F.normalize(class_center, dim=1).t().cuda())
                sim_s = torch.mm(F.normalize(fs.detach().cuda(), dim=1), F.normalize(fs.detach().cuda(), dim=1).t())
                one_hot_target = torch.zeros(batch_size, 100).scatter_(1, new_target.cpu().view(-1, 1), 1).cuda()
                one_hot_shuffle = torch.zeros(batch_size, 100).scatter_(1, new_target[rand_idx].cpu().view(-1, 1),
                                                                        1).cuda()
                wnet_in = torch.cat((score, one_hot_target), dim=1).detach().cuda()
                w = weight_net(wnet_in)
                with torch.no_grad():
                    _, _, ft = model_t(x_light)
                    pt_mix, _, _ = model_t(x_mix)
                    pt_mix = F.softmax(pt_mix / args.kd_T, dim=1)
                    sim_t = torch.mm(F.normalize(ft.detach().cuda(), dim=1), F.normalize(ft.detach().cuda(), dim=1).t())
                l_ce = torch.mean(F.cross_entropy(ps, new_target, reduction='none') * w)
                l_ces = cross_entropy(ps_mix, one_hot_target) * l + cross_entropy(ps_mix, one_hot_shuffle) * (1 - l)
                l_sim = 0.5 * F.mse_loss(sim_s, sim_t)
                l_kd = 0.9 * args.kd_T * args.kd_T * F.kl_div(ps_mix, pt_mix, reduction='batchmean')
                loss = l_ce + l_kd + l_ces + l_sim
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_w.step()
        scheduler.step()
        scheduler_w.step()
        test(model=model_s, epoch=epoch, test_loader=test_loader, args=args)

if __name__ == '__main__':
    main()
    
