import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image
import torchvision


def create_imblanced_data(labels, imb_type='exp', imb_factor=0.01):
    """reimp code from https://github.com/dvlab-research/Parametric-Contrastive-Learning"""
    labels = np.array(labels)
    cats = set(labels)
    cat_num = len(cats)
    img_max = len(labels) // cat_num
    if imb_type == 'exp':
        img_num_per_cls = [int(img_max * imb_factor ** (i / (cat_num - 1.0))) for i in range(cat_num)]
    elif imb_type == 'step':
        half = cat_num // 2
        img_num_per_cls = [img_max] * half + [int(img_max * imb_factor)] * (cat_num - half)
    else:
        img_num_per_cls = [img_max] * cat_num

    indexs = []
    for i, num in zip(range(cat_num), img_num_per_cls):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        indexs.extend(idx[:num])
    return indexs


class ImbalanceCIFAR10(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            sample1 = self.transform[0](img)
            sample2 = self.transform[1](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [sample1, sample2], target


from lumo.proc.path import cache_dir

cifar10 = torchvision.datasets.CIFAR100(cache_dir(), download=True)
index = create_imblanced_data(cifar10.targets, '', 0.01)
cifar10 = ImbalanceCIFAR10(cache_dir(), '')
print(len(cifar10.targets))
print(len(index))
