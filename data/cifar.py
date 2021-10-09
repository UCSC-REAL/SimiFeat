from __future__ import print_function

import os
import os.path
import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, noisify, noisify_instance, multiclass_noisify


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
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

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0, noise_file=None, synthetic=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar10'
        self.noise_type = noise_type
        self.nb_classes = 10
        self.noise_file = noise_file
        idx_each_class_noisy = [[] for i in range(10)]
        if download:
            self.download()

        if not self._check_integrity():
            self.download()
            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.noise_prior = None
            self.noise_or_not = None
            # if noise_type is not None:
            if noise_type != 'clean':
                # noisify train data
                if noise_type in ['symmetric', 'pairflip']:
                    self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                              train_labels=self.train_labels,
                                                                              noise_type=noise_type,
                                                                              noise_rate=noise_rate,
                                                                              random_state=random_state,
                                                                              nb_classes=self.nb_classes)
                    self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                    _train_labels = [i[0] for i in self.train_labels]
                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
                elif noise_type == 'instance':
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data,
                                                                                       self.train_labels,
                                                                                       noise_rate=noise_rate)
                    print('over all noise rate is ', self.actual_noise_rate)
                    # self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                    # self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                    # _train_labels=[i[0] for i in self.train_labels]
                    for i in range(len(self.train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                elif noise_type == 'manual':  # manual noise
                    # load noise label
                    train_noisy_labels = self.load_label()
                    self.train_noisy_labels = train_noisy_labels.numpy().tolist()
                    print(f'noisy labels loaded from {self.noise_file}')

                    for i in range(len(self.train_noisy_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    self.actual_noise_rate = np.sum(self.noise_or_not) / 50000
                    print('over all noise rate is ', self.actual_noise_rate)
                elif noise_type == 'rand_label':  # random labels
                    idx = np.arange(50000)
                    np.random.shuffle(idx)
                    self.train_noisy_labels = list(np.array(self.train_labels)[idx])
                    print(f'Use random labels: {self.train_noisy_labels[:10]}', flush=True)
                    # load noise label
                    # train_noisy_labels = self.load_label_human_new()
                    # self.train_noisy_labels = train_noisy_labels.tolist().copy()
                    # print(f'noisy labels loaded from {self.noise_file}')

                    # for i in range(len(self.train_noisy_labels)):
                    #     idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    # class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    # self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                    # print(f'The noisy data ratio in each class is {self.noise_prior}')
                    # self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                    # self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                    # print('over all noise rate is ', self.actual_noise_rate)
                else:
                    # load noise label
                    train_noisy_labels = self.load_label_human_new()
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    print(f'noisy labels loaded from {self.noise_file}')

                    if synthetic:
                        T = np.zeros((self.nb_classes, self.nb_classes))
                        for i in range(len(self.train_noisy_labels)):
                            T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                        T = T / np.sum(T, axis=1)
                        print(f'Noise transition matrix is \n{T}')
                        train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                                                random_state=random_state)  # np.random.randint(1,10086)
                        self.train_noisy_labels = train_noisy_labels.tolist()
                        T = np.zeros((self.nb_classes, self.nb_classes))
                        for i in range(len(self.train_noisy_labels)):
                            T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                        T = T / np.sum(T, axis=1)
                        print(f'New synthetic noise transition matrix is \n{T}')

                    for i in range(len(self.train_noisy_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    self.actual_noise_rate = np.sum(self.noise_or_not) / 50000
                    print('over all noise rate is ', self.actual_noise_rate)

            imbalance = False
            if imbalance:
                np.random.seed(1)
                img_num_list = self.get_img_num_per_cls(self.nb_classes, imb_type='step', imb_factor=0.1)
                self.gen_imbalanced_data(img_num_list)
                print(f'img_num_list is {img_num_list}')


        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label_human_new(self):
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0
                print(f'Loaded {self.noise_type} from {self.noise_file}.')
                print(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')

    def load_label(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        # NOTE presently only use for load manual training label
        # noise_label = torch.load(self.noise_type)   # f'../../{self.noise_type}'
        # noise_label = torch.load(f'noise_label/cifar-10/{self.noise_type}')
        assert self.noise_file != 'None'
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                assert torch.sum(torch.tensor(
                    self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            if "clean_label" in noise_label.keys() and 'raw_index' in noise_label.keys():
                assert torch.sum(
                    torch.tensor(self.train_labels)[noise_label['raw_index']] != noise_label['clean_label']) == 0
                noise_level = torch.sum(noise_label['clean_label'] == noise_label['noisy_label']) * 1.0 / (
                noise_label['clean_label'].shape[0])
                print(f'the overall noise level is {noise_level}')
                self.train_data = self.train_data[noise_label['raw_index']]
            return noise_label['noise_label_train'].view(-1).long() if 'noise_label_train' in noise_label.keys() else \
            noise_label['noisy_label'].view(-1).long()  # % 10

        else:
            return noise_label.view(-1).long()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.train_data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** ((cls_num - 1.0 - cls_idx) / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))

        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        idx_each_class_noisy = [[] for _ in range(10)]
        new_data = []
        new_targets = []
        new_noisy_label = []
        targets_np = np.array(self.train_labels, dtype=np.int64)
        train_noisy_labels = np.array(self.train_noisy_labels, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            reshape_times = the_img_num // idx.shape[0] + 1
            selec_idx = idx.repeat(reshape_times)[:the_img_num]
            new_data.append(self.train_data[selec_idx, ...])
            new_noisy_label += train_noisy_labels[selec_idx].tolist()
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.train_data = new_data
        self.train_labels = new_targets
        self.train_noisy_labels = new_noisy_label
        for i in range(len(self.train_noisy_labels)):
            idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
        class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
        self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
        print(f'The noisy data ratio in each class is {self.noise_prior}')
        # import pdb
        # pdb.set_trace()
        self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
        self.actual_noise_rate = np.sum(self.noise_or_not) / self.noise_or_not.shape[0]
        print('over all noise rate is ', self.actual_noise_rate)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.nb_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

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

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

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

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0, noise_file=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar100'
        self.noise_type = noise_type
        self.nb_classes = 100
        self.noise_file = noise_file
        idx_each_class_noisy = [[] for i in range(100)]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            # if noise_type is not None:
            #     # noisify train data
            #     self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
            #     self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
            #     self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            #     _train_labels=[i[0] for i in self.train_labels]
            #     for i in range(len(_train_labels)):
            #         idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
            #     class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
            #     self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            #     print(f'The noisy data ratio in each class is {self.noise_prior}')
            #     self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(_train_labels)
            if noise_type != 'clean':
                # noisify train data
                if noise_type in ['symmetric', 'pairflip']:
                    self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                              train_labels=self.train_labels,
                                                                              noise_type=noise_type,
                                                                              noise_rate=noise_rate,
                                                                              random_state=random_state,
                                                                              nb_classes=self.nb_classes)
                    self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                    _train_labels = [i[0] for i in self.train_labels]
                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
                elif noise_type == 'instance':
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data,
                                                                                       self.train_labels,
                                                                                       noise_rate=noise_rate)
                    print('over all noise rate is ', self.actual_noise_rate)
                    # self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                    # self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                    # _train_labels=[i[0] for i in self.train_labels]
                    for i in range(len(self.train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                elif noise_type == 'manual':  # manual noise
                    # load noise label
                    train_noisy_labels = self.load_label()
                    self.train_noisy_labels = train_noisy_labels.numpy().tolist()
                    print(f'noisy labels loaded from {self.noise_file}')

                    for i in range(len(self.train_noisy_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                else:  # random labels
                    train_noisy_labels = self.load_label_human_new()
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    print(f'noisy labels loaded from {self.noise_file}')

                    for i in range(len(self.train_noisy_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    self.actual_noise_rate = np.sum(self.noise_or_not) / 50000
                    print('over all noise rate is ', self.actual_noise_rate)

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        # NOTE presently only use for load manual training label
        # noise_label = torch.load(self.noise_type)   # f'../../{self.noise_type}'
        # noise_label = torch.load(f'noise_label/cifar-10/{self.noise_type}')
        assert self.noise_file != 'None'
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                assert torch.sum(torch.tensor(
                    self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            if "clean_label" in noise_label.keys() and 'raw_index' in noise_label.keys():
                assert torch.sum(
                    torch.tensor(self.train_labels)[noise_label['raw_index']] != noise_label['clean_label']) == 0
                noise_level = torch.sum(noise_label['clean_label'] == noise_label['noisy_label']) * 1.0 / (
                    noise_label['clean_label'].shape[0])
                print(f'the overall noise level is {noise_level}')
                self.train_data = self.train_data[noise_label['raw_index']]
            return noise_label['noise_label_train'].view(-1).long() if 'noise_label_train' in noise_label.keys() else \
                noise_label['noisy_label'].view(-1).long()  # % 10

        else:
            return noise_label.view(-1).long()  # % 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            # if self.noise_type is not None:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

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

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def load_label_human_new(self):
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_file)
        # import pdb
        # pdb.set_trace()
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0
                print(f'Loaded {self.noise_type} from {self.noise_file}.')
                print(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')
