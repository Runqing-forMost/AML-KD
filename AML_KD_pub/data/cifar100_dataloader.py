from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, conf=None, root_dir=None, transform=None, mode='test'):

        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        if self.mode == 'test':
            test_dic = unpickle('%s/test' % root_dir)
            self.test_data = test_dic['data']
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
            self.test_label = test_dic['fine_labels']

        else:
            train_dic = unpickle('%s/train' % root_dir)
            train_data = train_dic['data']
            train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if self.mode == 'all':
                self.train_data = train_data
                self.train_label = train_label
            else:
                if self.mode == "labeled":
                    pred_idx = conf.nonzero()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - conf).nonzero()

                # print(pred_idx.shape)
                self.train_data = train_data[pred_idx].squeeze()
                self.train_label = [train_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            # print(self.train_data.shape)
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target
        elif self.mode == 'unlabeled':

            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1
        elif self.mode == 'all':
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR100_DATALOADER():
    def __init__(self,  batch_size, num_workers, root_dir):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

    def run(self, mode, conf=None):
        if mode == 'train':
            labeled_dataset = cifar_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            conf=conf)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              conf=conf)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', conf=conf)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'warm-up':
            all_dataset = cifar_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'eval-train':
            eval_dataset = cifar_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all")
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
