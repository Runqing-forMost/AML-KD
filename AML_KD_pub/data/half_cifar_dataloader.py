from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torchvision.transforms as transforms
import pickle
import random
import torch
import torch.utils.data as data
import json


def tensor2PIL(tensor):  # 将tensor-> PIL
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True,
                 light_transform=None, heavy_transform=None, test_transform=None, download=False, mode='train',
                 noise_mode='None', noise_ratio=0.2, noise_file='', t_file='', s_file='', teacher=True):

        super(CIFAR10, self).__init__(root)
        self.light_transform = light_transform
        self.heavy_transform = heavy_transform
        self.test_transform = test_transform
        self.train = train  # training set or test set
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.noise_file = noise_file
        self.t_file = t_file
        self.s_file = s_file
        self.noise_mode = noise_mode
        self.noise_ratio = noise_ratio
        self.teacher = teacher  # whether is teacher

        if download:
            raise ValueError('cannot download.')
            exit()

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.teacher and self.train:  # if teacher mode
            if os.path.exists(self.t_file):
                t_idx = json.load(open(self.t_file, "r"))
            else:
                idx = list(range(50000))
                random.shuffle(idx)
                t_idx = idx[:25000]
                s_idx = idx[25000:]
                json.dump(t_idx, open(self.t_file, "w"))
                json.dump(s_idx, open(self.s_file, "w"))
            t_idx = np.array(t_idx)
            target = np.array(self.targets)
            self.targets = target[t_idx].tolist()
            self.data = self.data[t_idx]

        elif self.teacher is False and self.train:  # if student
            if os.path.exists(self.s_file):
                s_idx = json.load(open(self.s_file, "r"))
            else:
                idx = list(range(50000))
                random.shuffle(idx)
                t_idx = idx[:25000]
                s_idx = idx[25000:]
                json.dump(t_idx, open(self.t_file, "w"))
                json.dump(s_idx, open(self.s_file, "w"))
            s_idx_ = np.array(s_idx)
            target = np.array(self.targets)

            self.groud_truth = target[s_idx_].tolist()

            self.targets = target[s_idx_].tolist()
            self.data = self.data[s_idx_]

            # load noise file / teacher or student
            if self.noise_mode and self.noise_ratio > 0:
                if os.path.exists(self.noise_file):
                    noise_label = json.load(open(self.noise_file, "r"))
                else:  # inject noise
                    noise_label = []
                    idx = list(range(25000))
                    random.shuffle(idx)
                    num_noise = int(self.noise_ratio * 25000)
                    noise_idx = idx[:num_noise]
                    for i in range(25000):
                        if i in noise_idx:
                            if noise_mode == 'sym':
                                noiselabel = random.randint(0, 99)
                                noise_label.append(noiselabel)
                            elif noise_mode == 'asym':
                                noiselabel = (self.targets[i] + 1) % 100
                                noise_label.append(noiselabel)
                        else:
                            noise_label.append(self.targets[i])
                    print("save noisy labels to %s ..." % noise_file)
                    json.dump(noise_label, open(noise_file, "w"))

                self.targets = noise_label
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]

        img = Image.fromarray(img)
        if self.train:
            if self.teacher is False:
                gt = self.groud_truth[index]
                img1 = self.light_transform(img)

                img2 = self.heavy_transform(img)

                return img1, img2, target, index, gt
            else:
                img1 = self.light_transform(img)

                img2 = self.heavy_transform(img)
                return img1, img2, target
        else:

            img1 = self.test_transform(img)
            return img1, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
